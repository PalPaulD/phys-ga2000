import numpy as np

def get_bits(number):
    """For a NumPy quantity, return bit representation
    
    Inputs:
    ------
    number : NumPy value
        value to convert into list of bits
        
    Returns:
    -------
    bits : list
       list of 0 and 1 values, highest to lowest significance
    """
    bytes = number.tobytes()
    bits = []
    for byte in bytes:
        bits = bits + np.flip(np.unpackbits(np.uint8(byte)), np.uint8(0)).tolist()
    return list(reversed(bits))

bits = get_bits(np.float32(100.98763))

print(bits)
print('Sign: {}'.format(bits[0]))
print('Exponent: {}'.format(bits[1:9]))
print('Mantissa: {}'.format(bits[9:]))