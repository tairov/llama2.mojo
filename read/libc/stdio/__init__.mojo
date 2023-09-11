from memory.unsafe import Pointer


@value
@register_passable("trivial")
struct FILE:
    pass


# ===--------------------------------------------------------------------===#
# clearerr
# ===-----------------------------------------------------------------------===#


fn clearerr(arg: Pointer[FILE]) -> UInt8:
    return external_call["clearerr", UInt8, Pointer[FILE]](arg)


# ===--------------------------------------------------------------------===#
# fclose
# ===-----------------------------------------------------------------------===#


fn fclose(arg: Pointer[FILE]) -> Int32:
    return external_call["fclose", Int32](arg)


# ===--------------------------------------------------------------------===#
# feof
# ===-----------------------------------------------------------------------===#


fn feof(arg: Pointer[FILE]) -> Int32:
    return external_call["feof", Int32, Pointer[FILE]](arg)


# ===--------------------------------------------------------------------===#
# ferror
# ===-----------------------------------------------------------------------===#


fn ferror(arg: Pointer[FILE]) -> Int32:
    return external_call["ferror", Int32, Pointer[FILE]](arg)


# ===--------------------------------------------------------------------===#
# fflush
# ===-----------------------------------------------------------------------===#


fn fflush(arg: Pointer[FILE]) -> Int32:
    return external_call["fflush", Int32, Pointer[FILE]](arg)


# ===--------------------------------------------------------------------===#
# fgetc
# ===-----------------------------------------------------------------------===#


fn fgetc(arg: Pointer[FILE]) -> Int32:
    return external_call["fgetc", Int32, Pointer[FILE]](arg)


# ===--------------------------------------------------------------------===#
# fopen
# ===-----------------------------------------------------------------------===#


fn fopen(__filename: Pointer[UInt8], __mode: Pointer[UInt8]) -> Pointer[FILE]:
    return external_call["fopen", Pointer[FILE], Pointer[UInt8], Pointer[UInt8]](
        __filename, __mode
    )


# ===--------------------------------------------------------------------===#
# fread
# ===-----------------------------------------------------------------------===#


fn fread(
    __ptr: Pointer[UInt8], __size: UInt64, __nitems: UInt64, __stream: Pointer[FILE]
) -> UInt64:
    return external_call[
        "fread", UInt64, Pointer[UInt8], UInt64, UInt64, Pointer[FILE]
    ](__ptr, __size, __nitems, __stream)
