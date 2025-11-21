layouts
layouttensors

-- seems that only static allocation and statically creation needed

ndbuffer - can't allocate simple "slice" etc on top of memory, also it requires dims and dimlists 

spans - need LegacyUnsafePointer to initialize