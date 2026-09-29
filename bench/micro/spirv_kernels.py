"""Hand-assembled SPIR-V (OpenCL kernel execution model) for the bongo probe.

The reference box has no SYCL/DPC++ compiler installed, so the two micro
kernels the Level-Zero submission probe needs are assembled as SPIR-V words
here.  Everything emitted was accepted by the Intel IGC compiler inside NEO
(module creation returns ZE_RESULT_SUCCESS).

Kernels:

  k_store(__global volatile int *flag, int value)
      flag[0] = value;            (optionally followed by a memory fence)
  k_wait(__global volatile int *flag, int target)
      while (flag[0] != target) {}
  k_doorbell(...)   bounded doorbell, used by the U5 host->device probe
  k_empty()
"""
import struct

MAGIC = 0x07230203
# opcodes
OpCapability, OpMemoryModel, OpEntryPoint, OpSource = 17, 3, 15, 3
OpTypeVoid, OpTypeBool, OpTypeInt, OpTypePointer, OpTypeFunction = 19, 20, 21, 32, 33
OpConstant, OpFunction, OpFunctionParameter, OpLabel = 43, 54, 55, 248
OpStore, OpLoad, OpReturn, OpFunctionEnd, OpBranch = 62, 61, 253, 56, 249
OpBranchConditional, OpLoopMerge, OpINotEqual, OpIEqual = 250, 246, 171, 170
OpMemoryBarrier, OpAtomicStore = 224, 228
OpVariable, OpISub, OpIEqual, OpLogicalOr = 59, 130, 170, 166
# capabilities
Capability_Addresses, Capability_Kernel = 4, 6
# storage classes (SPIR-V spec 2.16)
StorageClass_CrossWorkgroup = 5
# memory semantics (OpenCL)
Semantics_SequentiallyConsistent = 0x10
Semantics_UniformMemory = 0x40
Semantics_CrossWorkgroupMemory = 0x200
# scopes (SPIR-V spec 3.28: CrossDevice 0, Device 1, Workgroup 2)
Scope_CrossDevice, Scope_Device, Scope_Workgroup = 0, 1, 2

STORE_MODES = ("plain", "volatile", "fence_uniform_crossdevice",
               "fence_crosswg_crossdevice", "fence_uniform_device",
               "atomic_crossdevice", "atomic_device")


class Asm:
    """Tiny SPIR-V word stream builder."""

    def __init__(self):
        # SPIR-V 1.2, generator magic 0, bound patched in finish(), schema 0
        self.words = [MAGIC, 0x00010200, 0, 1, 0]
        self.next = 1

    def id(self):
        i = self.next
        self.next += 1
        return i

    def op(self, opcode, *operands):
        self.words.append(((len(operands) + 1) << 16) | opcode)
        self.words.extend(operands)

    def op_entry_point(self, model, entry, name):
        b = name + b"\x00"
        pad = (-len(b)) % 4
        name_words = list(struct.unpack("<%dI" % ((len(b) + pad) // 4), b + b"\x00" * pad))
        self.op(OpEntryPoint, model, entry, *name_words)

    def finish(self):
        w = list(self.words)
        w[3] = self.next
        return struct.pack("<%dI" % len(w), *w)


def module_store(kernel_name=b"k_store", mode="volatile"):
    """__kernel void k_store(__global volatile int *flag, int value) { flag[0] = value; }"""
    assert mode in STORE_MODES, mode
    a = Asm()
    a.op(OpCapability, Capability_Kernel)
    a.op(OpCapability, Capability_Addresses)
    a.op(OpMemoryModel, 0, StorageClass_CrossWorkgroup)  # OpenCL, CrossWorkgroup
    void, int32, uint32 = a.id(), a.id(), a.id()
    ptr = a.id()
    fnty = a.id()
    main, pflag, pval = a.id(), a.id(), a.id()
    body = a.id()
    scope_cross, scope_dev = a.id(), a.id()
    sem_uniform, sem_crosswg = a.id(), a.id()
    a.op_entry_point(6, main, kernel_name)
    a.op(OpSource, 2, 200)  # OpenCL_C, version 2.0
    a.op(OpTypeVoid, void)
    a.op(OpTypeInt, int32, 32, 1)
    a.op(OpTypeInt, uint32, 32, 0)
    a.op(OpTypePointer, ptr, StorageClass_CrossWorkgroup, int32)
    a.op(OpTypeFunction, fnty, void, ptr, int32)
    a.op(OpConstant, uint32, scope_cross, Scope_CrossDevice)
    a.op(OpConstant, uint32, scope_dev, Scope_Device)
    a.op(OpConstant, uint32, sem_uniform,
         Semantics_UniformMemory | Semantics_SequentiallyConsistent)
    a.op(OpConstant, uint32, sem_crosswg,
         Semantics_CrossWorkgroupMemory | Semantics_SequentiallyConsistent)
    a.op(OpFunction, void, main, 0, fnty)
    a.op(OpFunctionParameter, ptr, pflag)
    a.op(OpFunctionParameter, int32, pval)
    a.op(OpLabel, body)
    if mode.startswith("atomic"):
        scope = scope_cross if mode == "atomic_crossdevice" else scope_dev
        a.op(OpAtomicStore, pflag, scope, sem_uniform, pval)
    else:
        access = 0 if mode == "plain" else 1  # MemoryAccess: Volatile
        a.op(OpStore, pflag, pval, access)
        if mode == "fence_uniform_crossdevice":
            a.op(OpMemoryBarrier, scope_cross, sem_uniform)
        elif mode == "fence_crosswg_crossdevice":
            a.op(OpMemoryBarrier, scope_cross, sem_crosswg)
        elif mode == "fence_uniform_device":
            a.op(OpMemoryBarrier, scope_dev, sem_uniform)
    a.op(OpReturn)
    a.op(OpFunctionEnd)
    return a.finish()


def module_wait(kernel_name=b"k_wait"):
    """__kernel void k_wait(__global volatile int *flag, int target) { while (flag[0] != target) {} }"""
    a = Asm()
    a.op(OpCapability, Capability_Kernel)
    a.op(OpCapability, Capability_Addresses)
    a.op(OpMemoryModel, 0, StorageClass_CrossWorkgroup)
    void, bool_, int32 = a.id(), a.id(), a.id()
    ptr, fnty = a.id(), a.id()
    main, pflag, ptgt = a.id(), a.id(), a.id()
    entry, loop, body, cont, merge = a.id(), a.id(), a.id(), a.id(), a.id()
    cur, ne = a.id(), a.id()
    a.op_entry_point(6, main, kernel_name)
    a.op(OpSource, 2, 200)
    a.op(OpTypeVoid, void)
    a.op(OpTypeBool, bool_)
    a.op(OpTypeInt, int32, 32, 1)
    a.op(OpTypePointer, ptr, StorageClass_CrossWorkgroup, int32)
    a.op(OpTypeFunction, fnty, void, ptr, int32)
    a.op(OpFunction, void, main, 0, fnty)
    a.op(OpFunctionParameter, ptr, pflag)
    a.op(OpFunctionParameter, int32, ptgt)
    a.op(OpLabel, entry)
    a.op(OpBranch, loop)
    a.op(OpLabel, loop)
    a.op(OpLoad, int32, cur, pflag, 1)  # Volatile
    a.op(OpINotEqual, bool_, ne, cur, ptgt)
    a.op(OpLoopMerge, merge, cont, 0)
    a.op(OpBranchConditional, ne, body, merge)
    a.op(OpLabel, body)
    a.op(OpBranch, cont)
    a.op(OpLabel, cont)
    a.op(OpBranch, loop)
    a.op(OpLabel, merge)
    a.op(OpReturn)
    a.op(OpFunctionEnd)
    return a.finish()


def module_doorbell_bounded(kernel_name=b"k_doorbell", budget=100_000_000):
    """Bounded device doorbell (never spins forever, so a missing host release
    cannot wedge the GPU):

        __kernel void k_doorbell(__global volatile int *flag, int value,
                                 __global volatile int *ack, __global int *result) {
            flag[0] = value;
            for (unsigned i = BUDGET; i != 0; --i)
                if (ack[0] != 0) break;      // volatile read
            result[0] = ack[0];              // 0 = never saw the host, 1 = saw it
        }
    """
    a = Asm()
    a.op(OpCapability, Capability_Kernel)
    a.op(OpCapability, Capability_Addresses)
    a.op(OpMemoryModel, 0, StorageClass_CrossWorkgroup)
    void, bool_, int32, uint32 = a.id(), a.id(), a.id(), a.id()
    ptr_i32, ptr_u32, fnty = a.id(), a.id(), a.id()
    main, pflag, pval, pack, presult = a.id(), a.id(), a.id(), a.id(), a.id()
    body0, loop, cont, merge = a.id(), a.id(), a.id(), a.id()
    i_var = a.id()
    iv, dec, av, seen, exhausted, done = a.id(), a.id(), a.id(), a.id(), a.id(), a.id()
    zero_i32, zero_u32, one_u32, budget_c = a.id(), a.id(), a.id(), a.id()
    a.op_entry_point(6, main, kernel_name)
    a.op(OpSource, 2, 200)
    a.op(OpTypeVoid, void)
    a.op(OpTypeBool, bool_)
    a.op(OpTypeInt, int32, 32, 1)
    a.op(OpTypeInt, uint32, 32, 0)
    a.op(OpTypePointer, ptr_i32, StorageClass_CrossWorkgroup, int32)
    a.op(OpTypePointer, ptr_u32, 7, uint32)  # StorageClass Function
    a.op(OpTypeFunction, fnty, void, ptr_i32, int32, ptr_i32, ptr_i32)
    a.op(OpConstant, int32, zero_i32, 0)
    a.op(OpConstant, uint32, zero_u32, 0)
    a.op(OpConstant, uint32, one_u32, 1)
    a.op(OpConstant, uint32, budget_c, budget)
    a.op(OpFunction, void, main, 0, fnty)
    a.op(OpFunctionParameter, ptr_i32, pflag)
    a.op(OpFunctionParameter, int32, pval)
    a.op(OpFunctionParameter, ptr_i32, pack)
    a.op(OpFunctionParameter, ptr_i32, presult)
    a.op(OpLabel, body0)
    a.op(OpVariable, ptr_u32, i_var, 7)
    a.op(OpStore, i_var, budget_c)
    a.op(OpStore, pflag, pval, 1)  # Volatile publish
    a.op(OpBranch, loop)
    a.op(OpLabel, loop)
    a.op(OpLoad, uint32, iv, i_var)
    a.op(OpISub, uint32, dec, iv, one_u32)
    a.op(OpStore, i_var, dec)
    a.op(OpLoad, int32, av, pack, 1)  # Volatile ack read
    a.op(OpINotEqual, bool_, seen, av, zero_i32)
    a.op(OpIEqual, bool_, exhausted, dec, zero_u32)
    a.op(OpLogicalOr, bool_, done, seen, exhausted)
    a.op(OpLoopMerge, merge, cont, 0)
    a.op(OpBranchConditional, done, merge, cont)
    a.op(OpLabel, cont)
    a.op(OpBranch, loop)
    a.op(OpLabel, merge)
    a.op(OpStore, presult, av)
    a.op(OpReturn)
    a.op(OpFunctionEnd)
    return a.finish()


def module_empty(kernel_name=b"k_empty"):
    a = Asm()
    a.op(OpCapability, Capability_Kernel)
    a.op(OpMemoryModel, 0, StorageClass_CrossWorkgroup)
    void, fnty, main, body = a.id(), a.id(), a.id(), a.id()
    a.op_entry_point(6, main, kernel_name)
    a.op(OpTypeVoid, void)
    a.op(OpTypeFunction, fnty, void)
    a.op(OpFunction, void, main, 0, fnty)
    a.op(OpLabel, body)
    a.op(OpReturn)
    a.op(OpFunctionEnd)
    return a.finish()


def module_doorbell_bounded(kernel_name=b"k_doorbell", budget=100_000_000):
    """Bounded device doorbell (never spins forever, so a missing host release
    cannot wedge the GPU):

        __kernel void k_doorbell(__global volatile int *flag, int value,
                                 __global volatile int *ack, __global int *result) {
            flag[0] = value;
            for (unsigned i = BUDGET; i != 0; --i)
                if (ack[0] != 0) break;      // volatile read
            result[0] = ack[0];              // 0 = never saw the host, 1 = saw it
        }
    """
    a = Asm()
    a.op(OpCapability, Capability_Kernel)
    a.op(OpCapability, Capability_Addresses)
    a.op(OpMemoryModel, 0, StorageClass_CrossWorkgroup)
    void, bool_, int32, uint32 = a.id(), a.id(), a.id(), a.id()
    ptr_i32, ptr_u32, fnty = a.id(), a.id(), a.id()
    main, pflag, pval, pack, presult = a.id(), a.id(), a.id(), a.id(), a.id()
    body0, loop, cont, merge = a.id(), a.id(), a.id(), a.id()
    i_var = a.id()
    iv, dec, av, seen, exhausted, done = a.id(), a.id(), a.id(), a.id(), a.id(), a.id()
    zero_i32, zero_u32, one_u32, budget_c = a.id(), a.id(), a.id(), a.id()
    a.op_entry_point(6, main, kernel_name)
    a.op(OpSource, 2, 200)
    a.op(OpTypeVoid, void)
    a.op(OpTypeBool, bool_)
    a.op(OpTypeInt, int32, 32, 1)
    a.op(OpTypeInt, uint32, 32, 0)
    a.op(OpTypePointer, ptr_i32, StorageClass_CrossWorkgroup, int32)
    a.op(OpTypePointer, ptr_u32, 7, uint32)  # StorageClass Function
    a.op(OpTypeFunction, fnty, void, ptr_i32, int32, ptr_i32, ptr_i32)
    a.op(OpConstant, int32, zero_i32, 0)
    a.op(OpConstant, uint32, zero_u32, 0)
    a.op(OpConstant, uint32, one_u32, 1)
    a.op(OpConstant, uint32, budget_c, budget)
    a.op(OpFunction, void, main, 0, fnty)
    a.op(OpFunctionParameter, ptr_i32, pflag)
    a.op(OpFunctionParameter, int32, pval)
    a.op(OpFunctionParameter, ptr_i32, pack)
    a.op(OpFunctionParameter, ptr_i32, presult)
    a.op(OpLabel, body0)
    a.op(OpVariable, ptr_u32, i_var, 7)
    a.op(OpStore, i_var, budget_c)
    a.op(OpStore, pflag, pval, 1)  # Volatile publish
    a.op(OpBranch, loop)
    a.op(OpLabel, loop)
    a.op(OpLoad, uint32, iv, i_var)
    a.op(OpISub, uint32, dec, iv, one_u32)
    a.op(OpStore, i_var, dec)
    a.op(OpLoad, int32, av, pack, 1)  # Volatile ack read
    a.op(OpINotEqual, bool_, seen, av, zero_i32)
    a.op(OpIEqual, bool_, exhausted, dec, zero_u32)
    a.op(OpLogicalOr, bool_, done, seen, exhausted)
    a.op(OpLoopMerge, merge, cont, 0)
    a.op(OpBranchConditional, done, merge, cont)
    a.op(OpLabel, cont)
    a.op(OpBranch, loop)
    a.op(OpLabel, merge)
    a.op(OpStore, presult, av)
    a.op(OpReturn)
    a.op(OpFunctionEnd)
    return a.finish()


def module_empty(kernel_name=b"k_empty"):
    a = Asm()
    a.op(OpCapability, Capability_Kernel)
    a.op(OpMemoryModel, 0, StorageClass_CrossWorkgroup)
    void, fnty, main, body = a.id(), a.id(), a.id(), a.id()
    a.op_entry_point(6, main, kernel_name)
    a.op(OpTypeVoid, void)
    a.op(OpTypeFunction, fnty, void)
    a.op(OpFunction, void, main, 0, fnty)
    a.op(OpLabel, body)
    a.op(OpReturn)
    a.op(OpFunctionEnd)
    return a.finish()


def module_doorbell(kernel_name=b"k_doorbell"):
    """__kernel void k_doorbell(__global volatile int *flag, int value, __global volatile int *ack) {
         flag[0] = value; while (ack[0] == 0) {} }"""
    a = Asm()
    a.op(OpCapability, Capability_Kernel)
    a.op(OpCapability, Capability_Addresses)
    a.op(OpMemoryModel, 0, StorageClass_CrossWorkgroup)
    void, bool_, int32 = a.id(), a.id(), a.id()
    ptr, fnty = a.id(), a.id()
    main, pflag, pval, pack = a.id(), a.id(), a.id(), a.id()
    body0, loop, body, cont, merge = a.id(), a.id(), a.id(), a.id(), a.id()
    cur, eq, zero = a.id(), a.id(), a.id()
    a.op_entry_point(6, main, kernel_name)
    a.op(OpSource, 2, 200)
    a.op(OpTypeVoid, void)
    a.op(OpTypeBool, bool_)
    a.op(OpTypeInt, int32, 32, 1)
    a.op(OpTypePointer, ptr, StorageClass_CrossWorkgroup, int32)
    a.op(OpTypeFunction, fnty, void, ptr, int32, ptr)
    a.op(OpConstant, int32, zero, 0)
    a.op(OpFunction, void, main, 0, fnty)
    a.op(OpFunctionParameter, ptr, pflag)
    a.op(OpFunctionParameter, int32, pval)
    a.op(OpFunctionParameter, ptr, pack)
    a.op(OpLabel, body0)
    a.op(OpStore, pflag, pval, 1)  # Volatile
    a.op(OpBranch, loop)
    a.op(OpLabel, loop)
    a.op(OpLoad, int32, cur, pack, 1)  # Volatile
    a.op(OpIEqual, bool_, eq, cur, zero)
    a.op(OpLoopMerge, merge, cont, 0)
    a.op(OpBranchConditional, eq, body, merge)
    a.op(OpLabel, body)
    a.op(OpBranch, cont)
    a.op(OpLabel, cont)
    a.op(OpBranch, loop)
    a.op(OpLabel, merge)
    a.op(OpReturn)
    a.op(OpFunctionEnd)
    return a.finish()


if __name__ == "__main__":
    for f in (module_empty, module_wait, module_doorbell_bounded):
        d = f()
        print(f.__name__, len(d), hex(struct.unpack("<I", d[:4])[0]))
    for m in STORE_MODES:
        d = module_store(mode=m)
        print("module_store", m, len(d))
