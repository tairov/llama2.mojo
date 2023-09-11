from memory.unsafe import Pointer


@value
@register_passable("trivial")
struct DIR:
    pass


@value
@register_passable("trivial")
struct dirent:
    var d_ino: UInt64
    var d_off: UInt64
    var d_reclen: UInt16
    var d_type: UInt8
    var d_name: Pointer[UInt8]


# ===--------------------------------------------------------------------===#
# closedir
# ===-----------------------------------------------------------------------===#


fn closedir(arg: Pointer[DIR]) -> Int32:
    return external_call["closedir", Int32, Pointer[DIR]](arg)


# ===--------------------------------------------------------------------===#
# opendir
# ===-----------------------------------------------------------------------===#


fn opendir(arg: Pointer[UInt8]) -> Pointer[DIR]:
    return external_call["opendir", Pointer[DIR], Pointer[UInt8]](arg)


# ===--------------------------------------------------------------------===#
# readdir
# ===-----------------------------------------------------------------------===#


fn readdir(arg: Pointer[DIR]) -> Pointer[dirent]:
    return external_call["readdir", Pointer[dirent], Pointer[DIR]](arg)


# ===--------------------------------------------------------------------===#
# fdopendir
# ===-----------------------------------------------------------------------===#


fn fdopendir(arg: Int32) -> DIR:
    return external_call["fdopendir", DIR](arg)


#
