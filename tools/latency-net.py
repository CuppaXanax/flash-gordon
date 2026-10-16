"""Payload echo with server service subtraction; clocks never cross hosts."""
import argparse
import json
import socket
import struct
import time


def exact(sock, count):
    data = bytearray()
    while len(data) < count:
        part = sock.recv(count - len(data))
        if not part:
            raise EOFError()
        data.extend(part)
    return data


def quantiles(values):
    v = sorted(values)
    return {"p50_us": v[len(v) // 2], "p99_us": v[int((len(v) - 1) * .99)],
            "min_us": v[0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["server", "client"])
    p.add_argument("--host", default="192.168.42.53")
    args = p.parse_args()
    if args.mode == "server":
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("0.0.0.0", 19552))
            listener.listen()
            print("ready", flush=True)
            for _ in range(4):
                conn, _ = listener.accept()
                with conn:
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    while True:
                        try:
                            count = struct.unpack("<I", exact(conn, 4))[0]
                        except EOFError:
                            break
                        payload = exact(conn, count)
                        start = time.perf_counter_ns()
                        conn.sendall(payload)
                        service = time.perf_counter_ns() - start
                        conn.sendall(struct.pack("<Q", service))
        return
    for count in [64, 20480, 40960, 51200]:
        samples, hop, service_samples = [], [], []
        with socket.create_connection((args.host, 19552), timeout=10) as conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(10)
            payload = bytes((i * 29 + 7) & 255 for i in range(count))
            message = struct.pack("<I", count) + payload
            for iteration in range(2100):
                start = time.perf_counter_ns()
                conn.sendall(message)
                returned = exact(conn, count)
                stop = time.perf_counter_ns()
                service = struct.unpack("<Q", exact(conn, 8))[0]
                if returned != payload:
                    raise RuntimeError("payload mismatch")
                if iteration >= 100:
                    samples.append((stop - start) / 1000)
                    hop.append((stop - start - service) / 2000)
                    service_samples.append(service / 1000)
        print(json.dumps({"bytes": count, "samples": len(samples),
                          "rtt": quantiles(samples), "rtt_minus_send_over_2": quantiles(hop),
                          "server_send": quantiles(service_samples),
                          "rtt_over_2": quantiles([x / 2 for x in samples]),
                          "exact": True}), flush=True)


if __name__ == "__main__":
    main()
