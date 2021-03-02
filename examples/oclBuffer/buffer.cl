__kernel void add_buff(
    __global unsigned char* A,
    __global unsigned char* B,
    unsigned char Add)
{
  // Get the index of the current element
  int i = get_global_id(0);

  // Do the operation
  B[i] = A[i] + Add;
}

__kernel void mult_buff(
    __global unsigned char* A,
    __global unsigned char* B,
    unsigned char Mult)
{
  // Get the index of the current element
  int i = get_global_id(0);

  // Do the operation
  B[i] = A[i] * Mult;
}

