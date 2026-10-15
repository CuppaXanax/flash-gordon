"""Instrument only the isolated microbenchmark build for same-batch kernel A/B."""
from pathlib import Path


def replace(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


vk = Path("src/vk.c")
test = Path("tests/test_fg_vk.c")
vk_source = vk.read_text()
test_source = test.read_text()
Path("candidate-vk.c").write_text(vk_source)
Path("candidate-test.c").write_text(test_source)
source = replace(vk_source, "fg_vk_kernel moe_decode_shared_add;", "fg_vk_kernel reference_gu,reference_dr;fg_vk_kernel moe_decode_shared_add;")
source = replace(source, 'c->moe_decode_gate_up=(fg_vk_kernel)', 'c->reference_gu=(fg_vk_kernel){.file="reference_gu.spv",.bindings=5,.push_bytes=32};c->reference_dr=(fg_vk_kernel){.file="reference_dr.spv",.bindings=5,.push_bytes=28};c->moe_decode_gate_up=(fg_vk_kernel)')
source = replace(source, "destroy_kernel(c,&c->moe_decode_shared_add);", "destroy_kernel(c,&c->reference_gu);destroy_kernel(c,&c->reference_dr);destroy_kernel(c,&c->moe_decode_shared_add);")
source = replace(source, "dispatch(c,&c->moe_decode_gate_up,bindings,&push,", 'dispatch(c,getenv("FG_EXPERT_REFERENCE")&&strcmp(getenv("FG_EXPERT_REFERENCE"),"1")==0?&c->reference_gu:&c->moe_decode_gate_up,bindings,&push,')
source = replace(source, "dispatch(c,cooked?&c->moe_decode_down_q8:&c->moe_decode_down_reduce,bindings,&push,", 'dispatch(c,getenv("FG_EXPERT_REFERENCE")&&strcmp(getenv("FG_EXPERT_REFERENCE"),"1")==0?&c->reference_dr:(cooked?&c->moe_decode_down_q8:&c->moe_decode_down_reduce),bindings,&push,')
source = replace(source, "&c->moe_decode_down_reduce),bindings,&push,groups,1u,1u,err);", '&c->moe_decode_down_reduce),bindings,&push,getenv("FG_EXPERT_REFERENCE")&&strcmp(getenv("FG_EXPERT_REFERENCE"),"1")==0?(q8_cooked?(output_width+7u)/8u:(output_width+15u)/16u):groups,1u,1u,err);')
vk.write_text(source)
start = test_source.index("static int run_expert_decode_pair_bench(")
end = test_source.index("static int test_expert_decode_pair_benchmark(", start)
bench = test_source[start:end]
bench = replace(bench, "ok=fg_vk_begin(context,&error)==FG_OK;", 'setenv("FG_EXPERT_REFERENCE",batch%2u?"1":"0",1);\n        ok=fg_vk_begin(context,&error)==FG_OK;')
bench = replace(bench, "uint32_t samples=iterations;", "uint32_t samples=iterations*2u;")
bench = replace(bench, "i<samples;i++){", 'i<samples;i++){\n        setenv("FG_EXPERT_REFERENCE",(i%4u==0u||i%4u==3u)?"1":"0",1);\n        rng=0x12345678u+(i/2u)*1234567u;')
bench = replace(bench, "(slot*3u+i)%SLOTS", "(slot*3u+i/2u)%SLOTS")
bench = replace(bench, 'rep=%u gu_us=', 'rep=%u reference=%s gu_us=')
bench = replace(bench, "gate_type,q8_down?8u:7u,i,gu_ms", 'gate_type,q8_down?8u:7u,i/2u,getenv("FG_EXPERT_REFERENCE"),gu_ms')
bench = replace(bench, "if(ok){\n        double gu_ms=gu_total", "if(0){\n        double gu_ms=gu_total")
test.write_text(test_source[:start] + bench + test_source[end:])
Path("vulkan/reference_gu.spv").write_bytes(Path("baseline/fg_moe_decode_gate_up.spv").read_bytes())
Path("vulkan/reference_dr.spv").write_bytes(Path("baseline/fg_moe_decode_down_reduce.spv").read_bytes())
