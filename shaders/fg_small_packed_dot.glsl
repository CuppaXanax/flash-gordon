uint small_packed_pair_dot(uint g,uint u,uint x,uint acc){
    [[unroll]] for(int shift=0;shift<32;shift+=8){
        uint weight=((g>>shift)&255u)|(((u>>shift)&255u)<<16u);
        acc+=weight*uint(bitfieldExtract(int(x),shift,8));
    }
    return acc;
}
ivec2 small_packed_pair_sum(uvec2 acc){
    // Eight Q5 x signed-Q8 products fit signed 16 bits (8*31*128=31744).
    // Negative low-lane sums borrow one from the high lane; restore it.
    ivec2 low=ivec2(acc<<16u)>>16;
    ivec2 high=(ivec2(acc)>>16)+ivec2(lessThan(low,ivec2(0)));
    return ivec2(low.x+low.y,high.x+high.y);
}
