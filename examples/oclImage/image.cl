__kernel void add_img(
    __read_only image2d_t A,
    __write_only image2d_t B,
    float Add)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

  int2 pixelcoord = (int2)(get_global_id(0), get_global_id(1));

  float4 pixelA = read_imagef(A, sampler, pixelcoord);

  pixelA += Add;

  write_imagef(B, pixelcoord, pixelA);
}

__kernel void mult_img(
    __read_only image2d_t A,
    __write_only image2d_t B,
    float Mult)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

  int2 pixelcoord = (int2)(get_global_id(0), get_global_id(1));

  float4 pixelA = read_imagef(A, sampler, pixelcoord);

  pixelA *= Mult;

  write_imagef(B, pixelcoord, pixelA);
}

