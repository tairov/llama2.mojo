from memory.unsafe import Pointer


fn strnlen(pointer: Pointer[UInt8]) -> Int:
    return external_call["strnlen", Int, Pointer[UInt8]](pointer)
