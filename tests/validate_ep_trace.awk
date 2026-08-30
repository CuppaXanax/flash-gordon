function value(name,    i,parts) {
    for (i = 1; i <= NF; i++) {
        split($i, parts, "=")
        if (parts[1] == name) {
            sub(/\r$/, "", parts[2])
            return parts[2]
        }
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
    sync1 = value("sync1_ms") + 0
    fire = value("fire_ms") + 0
    shared = value("shared_ms") + 0
    collect = value("collect_ms") + 0
    finish = value("finish_ms") + 0
    phases = sync1 + fire + shared + collect + finish
    difference = total - phases
    if (difference < 0) difference = -difference
    if (difference > 0.02) fail("layer " layer " phase sum differs by " difference " ms")
    submissions = value("submissions") + 0
    dispatches = value("dispatches") + 0
    if (submissions < 2 || submissions > 4) fail("layer " layer " submissions " submissions)
    if (dispatches < submissions) fail("layer " layer " dispatches " dispatches)
    layer_submissions += submissions
    layer_dispatches += dispatches
    layer_total += total
    sync1_total += sync1
    fire_total += fire
    shared_total += shared
    collect_total += collect
    finish_total += finish
    submission_layers[submissions]++
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
    expert_id_count = split(value("expert_ids"), expert_ids, ",")
    expert_rank_count = split(value("expert_ranks"), expert_ranks, ",")
    if (expert_id_count != 10 || expert_rank_count != 10) {
        fail("layer " layer " route trace does not contain ten expert IDs and ranks")
    } else {
        for (i = 1; i <= 10; i++) {
            expert_id = expert_ids[i]
            expert_rank = expert_ranks[i]
            if (expert_id !~ /^[0-9]+$/ || expert_id + 0 >= 512)
                fail("layer " layer " invalid expert ID " expert_id)
            else if (route_expert_seen[layer,expert_id]++)
                fail("layer " layer " duplicate expert ID " expert_id)
            if (expert_rank !~ /^[0-9]+$/ || expert_rank + 0 >= 8)
                fail("layer " layer " invalid expert rank " expert_rank)
            else {
                expert_rank += 0
                if (!group_has(layer, expert_rank))
                    fail("layer " layer " expert assigned outside group to rank " expert_rank)
                if (!has_bit(rank_mask[layer], expert_rank))
                    fail("layer " layer " expert rank missing from route mask " expert_rank)
                traced_rank_selected[layer,expert_rank]++
            }
        }
    }
    remote_routes += remote_count[layer]
    local_routes += local_count[layer]
    local_selections += local_selected[layer]
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
    rank_requests[rank]++
    rank_selections[rank] += worker_selected[layer,rank]
    rank_gpu_ms[rank] += value("gpu") + 0
    rank_reduce_ms[rank] += value("reduce") + 0
    rank_send_ms[rank] += value("send") + 0
    rank_total_ms[rank] += value("total") + 0
    worker_requests++
}

$0 ~ /TOKEN_PROFILE / && $0 !~ /TOKEN_PROFILE_KERNEL/ {
    if (value("rank") + 0 == 0 && value("kind") == "token" && value("token") + 0 == expected_token) {
        profile_count++
        profile_submissions = value("submissions") + 0
        profile_dispatches = value("dispatches") + 0
        profile_wall_ms = value("wall_ms") + 0
        profile_gpu_ms = value("gpu_ms") + 0
        profile_kernel_ms = value("kernel_ms") + 0
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
        if (traced_rank_selected[layer,0] != local_selected[layer])
            fail("layer " layer " traced/local selection mismatch")

        covered = local_selected[layer]
        for (rank = 0; rank < 8; rank++) {
            expected = has_bit(rank_mask[layer], rank)
            if (expected && !group_has(layer, rank))
                fail("layer " layer " routes outside expert group to rank " rank)
            if (rank == 0) continue
            observed = worker_seen[layer,rank] ? 1 : 0
            if (expected != observed)
                fail("layer " layer " worker presence mismatch rank " rank)
            if (observed) {
                covered += worker_selected[layer,rank]
                if (traced_rank_selected[layer,rank] != worker_selected[layer,rank])
                    fail("layer " layer " traced/worker selection mismatch rank " rank)
            }
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
    printf "EP_PHASE_SUM token=%u sync1_ms=%.3f fire_ms=%.3f shared_ms=%.3f collect_ms=%.3f finish_ms=%.3f nonlayer_ms=%.3f gpu_ms=%.3f kernel_ms=%.3f submit2_layers=%u submit3_layers=%u submit4_layers=%u\n", expected_token, sync1_total, fire_total, shared_total, collect_total, finish_total, profile_wall_ms-layer_total, profile_gpu_ms, profile_kernel_ms, submission_layers[2], submission_layers[3], submission_layers[4]
    printf "EP_ROUTE_SUM token=%u remote_routes=%u local_routes=%u local_selections=%u remote_selections=%u\n", expected_token, remote_routes, local_routes, local_selections, 480-local_selections
    for (rank = 1; rank < 8; rank++) printf "EP_WORKER_SUM token=%u rank=%u requests=%u selections=%u gpu_ms=%.3f reduce_ms=%.3f send_ms=%.3f total_ms=%.3f\n", expected_token, rank, rank_requests[rank], rank_selections[rank], rank_gpu_ms[rank], rank_reduce_ms[rank], rank_send_ms[rank], rank_total_ms[rank]
}