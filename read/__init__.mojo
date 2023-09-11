"""
This is a port of Zig's buffered reader.
See: https://github.com/ziglang/zig/blob/master/lib/std/io/buffered_reader.zig
The MIT License (Expat)

Copyright (c) Lukas Hermann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from utils.list import Dim
from math import min
from math.limit import max_finite
from memory import memcpy
from memory.buffer import Buffer
from memory.unsafe import Pointer, DTypePointer
from sys.info import sizeof
from utils.index import Index
from utils.vector import DynamicVector
import testing


from .libc.stdio import fopen, fread, fclose, FILE
from .libc.dirent import readdir, opendir, closedir, DIR, dirent
from .libc.string import strnlen

alias BUF_SIZE = 4096

# Types aliases
alias c_char = UInt8


fn to_char_ptr(s: String) -> Pointer[c_char]:
    """Only ASCII-based strings."""
    let ptr = Pointer[c_char]().alloc(len(s) + 1)
    for i in range(len(s)):
        ptr.store(i, ord(s[i]))
    ptr.store(len(s), ord("\0"))
    return ptr


struct File:
    var handle: Pointer[FILE]
    var fname: Pointer[c_char]
    var mode: Pointer[c_char]

    fn __init__(inout self, filename: String):
        let fname = to_char_ptr(filename)
        let mode = to_char_ptr("r")
        let handle = fopen(fname, mode)

        self.fname = fname
        self.mode = mode
        self.handle = handle

    fn __bool__(self) -> Bool:
        return self.handle.__bool__()

    fn __del__(owned self) raises:
        if self.handle:
            pass
            # let c = fclose(self.handle)
            # if c != 0:
            #     raise Error("Failed to close file")
        if self.fname:
            self.fname.free()
        if self.mode:
            self.mode.free()

    fn __moveinit__(inout self, owned other: Self):
        self.fname = other.fname
        self.mode = other.mode
        self.handle = other.handle
        other.handle = Pointer[FILE]()
        other.fname = Pointer[c_char]()
        other.mode = Pointer[c_char]()

    fn do_nothing(self):
        pass

    fn read[D: Dim](self, buffer: Buffer[D, DType.uint8]) raises -> Int:
        return fread(
            buffer.data.as_scalar_pointer(), sizeof[UInt8](), BUF_SIZE, self.handle
        ).to_int()


struct DirEntry:
    var _pointer: Pointer[dirent]
    var name: String

    fn __init__(inout self, pointer: Pointer[dirent]):
        self.name = String()
        if pointer:
            print("hit")
            let name_ptr = pointer.bitcast[UInt8]().offset(
                sizeof[UInt64]() * 2 + sizeof[UInt16]() + sizeof[UInt8]()
            )
            let name_len = strnlen(name_ptr)
            for i in range(name_len):
                self.name += chr(name_ptr.load(i).to_int())
        self._pointer = pointer


@value
@register_passable("trivial")
struct DirIter:
    var handle: Pointer[DIR]
    var data: Pointer[dirent]

    fn __iter__(inout self) -> Self:
        self.data = readdir(self.handle)

    fn __next__(self) raises -> Pointer[dirent]:
        return self.data

    fn __len__(self) -> Int:
        if self.handle and self.data:
            return 1
        return 0


struct Dir:
    var handle: Pointer[DIR]
    var path: Pointer[c_char]

    fn __init__(inout self, path: String):
        self.path = to_char_ptr(path)
        self.handle = opendir(self.path)

    fn __bool__(self) -> Bool:
        return self.handle.__bool__()

    fn __iter__(self) -> DirIter:
        return DirIter(self.handle, Pointer[dirent]())

    fn __del__(owned self) raises:
        let c = closedir(self.handle)
        if c != 0:
            raise Error("failed to close dir")
        self.path.free()

    fn do_nothing(self):
        pass


struct BufReader[BUF_SIZE: Int]:
    var unbuffered_reader: File
    var data: DTypePointer[DType.uint8]
    var end: Int
    var start: Int

    fn __init__(inout self, owned reader: File):
        self.unbuffered_reader = reader ^
        self.data = DTypePointer[DType.uint8]().alloc(BUF_SIZE)
        self.end = 0
        self.start = 0

    fn read[D: Dim](inout self, dest: Buffer[D, DType.uint8]) raises -> Int:
        var dest_index = 0
        let buf = Buffer[BUF_SIZE, DType.uint8](self.data)

        while dest_index < len(dest):
            let written = min(len(dest) - dest_index, self.end - self.start)
            memcpy(dest.data.offset(dest_index), self.data.offset(self.start), written)
            if written == 0:
                # buf empty, fill it
                let n = self.unbuffered_reader.read(buf)
                if n == 0:
                    # reading from the unbuffered stream returned nothing
                    # so we have nothing left to read.
                    return dest_index
                self.start = 0
                self.end = n
            self.start += written
            dest_index += written
        return len(dest)

    fn do_nothing(self):
        pass
