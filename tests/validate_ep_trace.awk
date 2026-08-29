function value(name,    i,parts) {
    for (i = 1; i <= NF; i++) {
        split($i, parts, "=")
        if (parts[1] == name) return parts[2]
    }
    return ""
}

function trace_layer(    i,result) {
    for (i = 1; i <= NF; i++) {
        if ($i ~ /^layer\[[0-9]+\]$/) {
            result = $i
            sub(/^layer\[/, "", result)
            sub(/\]$/, "", result)
            return result + 0
        }
    }
    return -1
}

function has_bit(mask, rank) { return int(mask / (2 ^ rank)) % 2 }
function group_has(layer, rank) {
    return rank == layer % 8 || rank == (layer + 1) % 8 || rank == (layer + 3) % 8 || rank == (layer + 5) % 8
}
function fail(message) {
    print "EP_TRACE_ERROR " message > "/dev/stderr"
    errors++
}

$0 ~ /EP_LAYER_TRACE / {
    current_token = value("token") + 0
    if (current_token != expected_token) next
    layer = value("layer") + 0
    if (layer < 0 || layer >= 48) { fail("invalid layer " layer); next }
    if (layer_seen[layer]++) { fail("duplicate layer trace " layer); next }
    if (value("status") + 0 != 0) fail("layer " layer " status " value("status"))
    total = value("total_ms") + 0
    phases = value("sync1_ms") + value("fire_ms") + value("shared_ms") + value("collect_ms") + value("finish_ms")
    difference = total - phases
    if (difference < 0) difference = -difference
    if (difference > 0.02) fail("layer " layer " phase sum differs by " difference " ms")
    submissions = value("submissions") + 0
    dispatches = value("dispatches") + 0
    if (submissions < 3 || submissions > 4) fail("layer " layer " submissions " submissions)
    if (dispatches < submissions) fail("layer " layer " dispatches " dispatches)
    layer_submissions += submissions
    layer_dispatches += dispatches
    layer_total += total
    if (total > maximum_layer_ms) { maximum_layer_ms = total; maximum_layer = layer }
    layer_count++
}

$0 ~ /EP_ROUTE_TRACE / {
    current_token = value("token") + 0
    if (current_token != expected_token) next
    layer = value("layer") + 0
    if (layer < 0 || layer >= 48) { fail("invalid route layer " layer); next }
    if (route_seen[layer]++) { fail("duplicate route trace " layer); next }
    route_count[layer] = value("routes") + 0
    remote_count[layer] = value("remotes") + 0
    local_count[layer] = value("local") + 0
    local_selected[layer] = value("local_selected") + 0
    selected_count[layer] = value("selected") + 0
    rank_mask[layer] = value("rank_mask") + 0
    routes++
}

$0 ~ /WORKER_EXPERT / {
    current_token = value("t") + 0
    if (current_token != expected_token) next
    layer = trace_layer()
    rank = value("rank") + 0
    if (layer < 0 || layer >= 48 || rank < 1 || rank >= 8) {
        fail("invalid worker trace layer=" layer " rank=" rank)
        next
    }
    if (worker_seen[layer,rank]++) fail("duplicate worker trace layer=" layer " rank=" rank)
    worker_selected[layer,rank] = value("sel") + 0
    worker_requests++
}

$0 ~ /TOKEN_PROFILE / && $0 !~ /TOKEN_PROFILE_KERNEL/ {
    if (value("rank") + 0 == 0 && value("kind") == "token" && value("token") + 0 == expected_token) {
        profile_count++
        profile_submissions = value("submissions") + 0
        profile_dispatches = value("dispatches") + 0
    }
}

END {
    if (layer_count != 48) fail("expected 48 layer traces, found " layer_count)
    if (routes != 48) fail("expected 48 route traces, found " routes)
    if (profile_count != 1) fail("expected one coordinator token profile, found " profile_count)

    for (layer = 0; layer < 48; layer++) {
        if (!layer_seen[layer]) fail("missing layer trace " layer)
        if (!route_seen[layer]) { fail("missing route trace " layer); continue }
        if (route_count[layer] < 1 || route_count[layer] > 4)
            fail("layer " layer " route count " route_count[layer])
        if (remote_count[layer] + local_count[layer] != route_count[layer])
            fail("layer " layer " route partition mismatch")
        if (local_count[layer] != has_bit(rank_mask[layer], 0))
            fail("layer " layer " local route/mask mismatch")
        if (selected_count[layer] != 10)
            fail("layer " layer " selected " selected_count[layer])

        covered = local_selected[layer]
        for (rank = 0; rank < 8; rank++) {
            expected = has_bit(rank_mask[layer], rank)
            if (expected && !group_has(layer, rank))
                fail("layer " layer " routes outside expert group to rank " rank)
            if (rank == 0) continue
            observed = worker_seen[layer,rank] ? 1 : 0
            if (expected != observed)
                fail("layer " layer " worker presence mismatch rank " rank)
            if (observed) covered += worker_selected[layer,rank]
        }
        if (covered != 10) fail("layer " layer " worker/local coverage " covered)
    }

    if (profile_count == 1) {
        if (profile_submissions - layer_submissions != 2)
            fail("non-layer submissions " (profile_submissions - layer_submissions))
        if (profile_dispatches - layer_dispatches != 2)
            fail("non-layer dispatches " (profile_dispatches - layer_dispatches))
    }

    if (errors) exit 1
    printf "EP_INTEGRATION_PASS token=%u layers=%u routes=%u worker_requests=%u layer_ms=%.3f max_layer=%u max_layer_ms=%.3f submissions=%u dispatches=%u\n", expected_token, layer_count, routes, worker_requests, layer_total, maximum_layer, maximum_layer_ms, profile_submissions, profile_dispatches
}