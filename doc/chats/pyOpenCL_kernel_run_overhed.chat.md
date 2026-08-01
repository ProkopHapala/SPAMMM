# USER

when using pyOpenCL we often fact problem that python have huge overhead, we an evaluate thousands or millions of pairwise interactions on GPU before python harness execute new kernel. How can we overcome that? We were desingning kernels wehere we run loop of multiple interations of moleculer dynamics inside one kernel call, ut this limits interaction in one workgroups, as there is not synchornization between workgroups. Can we somehow define some event-dependnet execution or somthihng, which will eliminate this limitalion of python harness. allowing to execute krenel many time s(e.g. 100x) with single python-side call ?

do not problam yet, just analyse the problem 

---

# Chat GPT 5.6 sol

You are facing **two separate limitations that are easy to conflate**:

1. the synchronization model of OpenCL kernels;
2. the cost of repeatedly submitting kernels from Python.

The first is fundamental. The second is largely an API/runtime issue and can, at least conceptually, be removed.

## 1. The MD algorithm does not require one giant kernel

Suppose one timestep is

[
\mathbf F_t = F(\mathbf x_t),
]

[
(\mathbf x_{t+1},\mathbf v_{t+1})
= I(\mathbf x_t,\mathbf v_t,\mathbf F_t).
]

Before evaluating (F(\mathbf x_{t+1})), every workgroup must see all positions written during the integration of timestep (t).

Inside one ordinary OpenCL kernel:

[
\texttt{barrier(...)}
]

only synchronizes work-items in the same workgroup. Even a global-memory fence controls memory ordering and visibility; it does not stop one workgroup until all other workgroups reach the same point.

But **completion of a kernel invocation is a valid device-wide synchronization point**: all its workgroups have finished before a dependent kernel begins. Therefore this sequence is perfectly correct:

```text
force(x[t])       -- global NDRange
integrate(x[t])   -- global NDRange
force(x[t+1])     -- global NDRange
integrate(x[t+1])
...
```

The real question is not whether you need kernel boundaries. You generally do. The question is:

> Must Python individually construct and enqueue every boundary?

No.

## 2. There are three different overheads

For very small kernels, the elapsed time contains approximately three components:

[
T_{\mathrm{step}}
=================

T_{\mathrm{Python}}
+
T_{\mathrm{driver\ enqueue}}
+
T_{\mathrm{device\ launch}}
+
T_{\mathrm{kernel}}.
]

They need different remedies.

### Python/PyOpenCL overhead

This includes:

* Python function dispatch;
* constructing tuples and argument lists;
* converting scalar arguments;
* allocating Python `Event` wrappers;
* possibly setting kernel arguments every time;
* accidentally obtaining a new `Kernel` object through repeated `program.kernel_name` lookup.

PyOpenCL explicitly notes that every `program.kernel_name` lookup creates a new kernel object and defeats enqueue-related caches attached to the kernel object. ([documen.tician.de][1])

### OpenCL driver submission overhead

Even a C program still has to call something equivalent to

```c
clEnqueueNDRangeKernel(...)
```

for each ordinary kernel launch. Removing Python does not remove this driver-level command construction.

### GPU launch latency

After submission, the device must create and schedule a new grid of workgroups. Even a command buffer cannot completely remove the logical kernel boundaries, because those boundaries provide the required global synchronization. A sufficiently smart driver may nevertheless reduce the dispatch overhead substantially.

This distinction matters:

* a small C helper eliminates mostly (T_{\mathrm{Python}});
* a command buffer can amortize both (T_{\mathrm{Python}}) and much of (T_{\mathrm{driver\ enqueue}});
* a persistent kernel removes repeated device launches, but then you must somehow implement a global barrier.

## 3. Ordinary OpenCL events do not form a loop

The other LLM is correct here.

An event is essentially a dependency token:

```text
kernel B may start after event from kernel A completes
```

It is not an executable continuation:

```text
when A completes, automatically enqueue A again
```

For a simple sequential MD trajectory, an **in-order command queue already expresses the ordering**. Explicit events add little unless you are using:

* an out-of-order queue;
* multiple queues;
* concurrent force components;
* asynchronous copies;
* more complicated dependency graphs.

OpenCL events can therefore describe

```text
A → B → C
```

but an ordinary event cannot describe

```text
repeat (A → B) 100 times
```

without someone submitting those commands.

Event callbacks would not fundamentally help. A callback runs host-side; a Python callback would re-enter Python and likely make latency and unpredictability worse.

## 4. Python can already enqueue ahead asynchronously

It is worth separating “100 Python calls” from “100 host-device synchronizations.”

The following conceptual loop:

```python
for i in range(100):
    force_kernel(...)
    integrate_kernel(...)
```

does not have to wait for each timestep. Provided there is no:

* `queue.finish()`;
* `event.wait()`;
* blocking readback;
* profiling-property access that waits for completion;

Python may enqueue commands ahead while the GPU processes previously submitted commands.

The execution can look like:

```text
CPU: enqueue 0, enqueue 1, enqueue 2, enqueue 3, ...
GPU:       run 0,     run 1,     run 2, ...
```

Thus the total time is closer to

[
T_{\mathrm{batch}}
\approx
\max\left(
N T_{\mathrm{host-submit}},
N(T_{\mathrm{launch}}+T_{\mathrm{kernel}})
\right),
]

not necessarily their sum.

The problem becomes serious when

[
T_{\mathrm{host-submit}}

>

T_{\mathrm{launch}}+T_{\mathrm{kernel}},
]

because the GPU command queue drains faster than Python fills it. Then bubbles appear between kernels.

## 5. A native C enqueue loop: one Python call, 100 OpenCL calls

The simplest conceptual separation is:

```text
Python
  |
  | one call
  v
small C/C++ function:
    for step = 0...99:
        clEnqueueNDRangeKernel(force)
        clEnqueueNDRangeKernel(integrate)
```

This would give:

* one Python call;
* no Python event objects per timestep;
* no repeated Python argument conversion;
* correct global synchronization through kernel boundaries;
* still 200 OpenCL driver enqueue calls;
* still 200 device kernel launches.

This is important because it tells us whether **Python itself** is the dominant problem.

It is also much more portable than advanced OpenCL features: it only requires ordinary OpenCL kernel enqueue operations.

Conceptually, this is a different solution from kernel fusion. The simulation remains globally decomposed into many kernels, but Python no longer participates in each timestep.

## 6. OpenCL command buffers are almost exactly what you described

The most direct answer to your question is `cl_khr_command_buffer`.

A command buffer separates:

1. constructing a command sequence;
2. submitting that previously constructed sequence.

You could record, once:

```text
force step 0
integrate step 0
force step 1
integrate step 1
...
force step 99
integrate step 99
```

and subsequently execute the entire sequence with one:

```c
clEnqueueCommandBufferKHR(...)
```

The kernels remain separate NDRanges, so every force or integration kernel may use the whole GPU. Kernel boundaries and recorded synchronization points maintain global correctness. The command-buffer specification explicitly describes recording and replay as a mechanism for reducing repeated command-construction and driver overhead. ([Khronos Registry][2])

The dependencies inside the command buffer can be encoded using:

* command-buffer synchronization points;
* recorded barriers;
* implicit ordering when recording against an in-order queue.

The command-buffer object is otherwise considered out-of-order, so dependencies must be represented correctly. ([Khronos Registry][2])

Conceptually:

```text
Python call
   |
   v
recorded command buffer
   |
   +-- force_0
   +-- integrate_0
   +-- force_1
   +-- integrate_1
   ...
   +-- integrate_99
```

This directly contradicts the overly pessimistic conclusion from the other LLM. Standard events alone cannot do it, but a command buffer can contain an **unrolled sequence of 100 globally synchronized kernel invocations**.

### It is not a device-side loop

A command buffer does not generally mean:

```c
for (i=0; i<100; i++)
    launch(kernel);
```

It means that 100 launches are recorded individually once.

That has consequences:

* the number of steps is effectively fixed when the command buffer is recorded;
* all buffer addresses and ordinary arguments are captured during recording;
* the recorded command sequence is static;
* different ping-pong arguments must be encoded into the sequence;
* adaptive branching is difficult.

The `cl_khr_command_buffer_mutable_dispatch` extension permits selected dispatch properties and arguments to be changed between command-buffer submissions, but not arbitrary data-dependent branching within a submission. ([Khronos Registry][3])

### State evolution does not require mutable arguments

This point is subtle but useful.

If every kernel receives pointers to persistent buffers such as:

```text
positions
velocities
forces
simulation_state
```

the **contents** of those buffers may change during execution. The command buffer only needs the memory-object handles to remain valid.

Therefore, a static recorded command sequence can evolve the state:

[
S_0\rightarrow S_1\rightarrow\cdots\rightarrow S_{100}
]

without changing its kernel arguments.

Ping-pong buffers require either:

```text
A→B, B→A, A→B, B→A...
```

to be recorded explicitly, or an indirection inside a state structure.

## 7. The practical command-buffer obstacle is support

There are two separate support questions:

### Driver support

The device must advertise `cl_khr_command_buffer`. OpenCL 3.0 does not imply that every optional modern feature is available. Device-side enqueue and many later capabilities are individually queryable optional features. ([Khronos Registry][4])

NVIDIA advertises OpenCL 3.0 support for current drivers, including the Ampere generation, but that alone does not establish support for command buffers. ([The Khronos Group][5])

For your RTX 3090, the decisive information is therefore its actual extension string and command-buffer capability queries under your installed driver.

### PyOpenCL support

As of the current PyOpenCL documentation and repository state, command-buffer support is not part of the normal exposed PyOpenCL API. The corresponding feature request remains open. ([GitHub][6])

So there are three layers:

```text
OpenCL specification: command buffers exist
GPU driver: may or may not implement them
PyOpenCL: currently does not expose them normally
```

Even if the driver supports the extension, a small native binding would probably be needed.

## 8. Device-side enqueue is more powerful than the other answer suggests

The other LLM's statement that device-side enqueue is limited to local workgroups is misleading.

OpenCL 2.0 device-side enqueue allows a parent kernel to enqueue child kernels with complete NDranges onto a device-side command queue. Dependencies can be represented using device-side events. The child kernels may therefore contain many workgroups and operate over the whole device. ([Khronos Registry][7])

Conceptually, one small scheduler kernel could do:

```text
enqueue global force kernel
enqueue global integration kernel dependent on force
enqueue next global force dependent on integration
...
```

This is true device-side command generation, not an intra-workgroup barrier.

It could also be data-dependent:

```text
if maximum displacement is large:
    enqueue neighbor-list rebuild
else:
    enqueue next force step
```

That is closer to the “event-dependent execution” you imagined.

However, device-side enqueue has serious qualifications:

* it is optional in modern OpenCL;
* many desktop GPU OpenCL drivers do not expose it;
* it requires an on-device command queue;
* child-kernel launch overhead may still be significant;
* implementation quality varies;
* it is much less broadly usable than ordinary OpenCL 1.2 functionality.

So it is conceptually valid, but probably not a robust portability foundation.

## 9. Persistent kernels and global software barriers

The other theoretical route is:

```text
launch one persistent kernel

for step = 0...99:
    evaluate interactions
    global_barrier()
    integrate
    global_barrier()
```

A global barrier can be approximated using atomic counters:

```text
workgroup leader increments counter
all workgroups spin until counter == number_of_workgroups
```

The notorious problem is residency.

Suppose the kernel launches 1000 workgroups, but the GPU can concurrently keep only 160 resident. The first 160 workgroups reach the barrier and spin. They occupy all available execution resources while waiting for the remaining 840 workgroups, which cannot be scheduled.

That is a deadlock.

It can only be made reasonably safe when the launch is restricted such that all workgroups are simultaneously resident:

[
N_{\mathrm{WG}}
\leq
N_{\mathrm{resident,WG}}.
]

But standard OpenCL does not provide a portable cooperative-grid-launch guarantee equivalent to saying:

> All these workgroups are resident together and may safely participate in a grid-wide barrier.

One may tune the kernel to roughly one persistent workgroup per compute unit, but then:

* occupancy is deliberately restricted;
* the mapping of workgroups to compute units is not guaranteed;
* resource usage changes residency;
* portability becomes poor;
* a driver/compiler change can invalidate the assumption.

So a software global barrier is not completely impossible, but it is an architectural gamble rather than a clean OpenCL solution.

## 10. Static versus adaptive trajectories

The shape of your MD control flow matters.

### Static batch

```text
perform exactly 100 timesteps
rebuild neighbors every 10 steps
write diagnostics at step 100
```

This is ideal for a command buffer. Everything can be unrolled and recorded.

### Mildly adaptive batch

```text
perform up to 100 timesteps
stop if converged
```

A static command buffer could still record 100 steps, while each kernel checks:

```c
if (!simulation_active)
    return;
```

This avoids host intervention but still launches no-op kernels after convergence.

### Strongly adaptive execution

```text
change timestep based on forces
rebuild neighbors when required
repeat constraint solver until converged
terminate trajectory on event
branch into different algorithms
```

A static command buffer cannot naturally express this as control flow. Then the possibilities become:

* process short chunks, for example 10–100 steps, and let the host inspect between chunks;
* always execute a superset of possible kernels and make unused kernels early-out;
* use device-side enqueue, if genuinely supported;
* redesign the numerical algorithm to reduce global decisions.

This distinction is more important than whether the host language is Python.

## 11. The actual landscape

| Mechanism                            | Python calls for 100 steps | Global multi-workgroup correctness | Removes driver enqueue overhead | Device-data-dependent control |
| ------------------------------------ | -------------------------: | ---------------------------------: | ------------------------------: | ----------------------------: |
| Ordinary PyOpenCL loop               |                    100–300 |                                Yes |                              No |                     Host only |
| Native C enqueue loop                |                          1 |                                Yes |                              No |                       Limited |
| OpenCL command buffer                |          1 after recording |                                Yes |               Largely amortized |                 Mostly static |
| Device-side enqueue                  |                          1 |                                Yes |           Host overhead removed |                           Yes |
| Persistent kernel, one WG            |                          1 |                                Yes |                             Yes |                           Yes |
| Persistent multi-WG + atomic barrier |                          1 |                 Conditional/unsafe |                             Yes |                           Yes |

## Main conclusion

The fundamental requirement is not:

> one kernel containing 100 MD iterations.

It is:

> one submitted execution graph containing 100 globally synchronized kernel invocations.

That distinction opens the correct solution space.

The closest conceptual match is:

```text
recorded command buffer =
    [force → integrate] × 100
```

It preserves full-device interaction at every step while requiring one submission from Python. A small native enqueue loop is the portable fallback that removes Python from the inner loop but retains individual OpenCL driver calls.

Events by themselves are only dependency edges. They cannot create repetition or data-dependent continuation. Device-side enqueue can do that in principle, but its practical support is much weaker.

[1]: https://documen.tician.de/pyopencl/runtime_program.html "OpenCL Runtime: Programs and Kernels - pyopencl 2026.1.2 documentation"
[2]: https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/cl_khr_command_buffer.html "cl_khr_command_buffer(3)"
[3]: https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/cl_khr_command_buffer_mutable_dispatch.html "cl_khr_command_buffer_mutable_dispatch(3)"
[4]: https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html?utm_source=chatgpt.com "The OpenCL™ Specification - Khronos Registry"
[5]: https://www.khronos.org/conformance/adopters/conformant-products/opencl?utm_source=chatgpt.com "OpenCL 3.0"
[6]: https://github.com/inducer/pyopencl/issues/820 "Support `cl_khr_command_buffer` · Issue #820 · inducer/pyopencl · GitHub"
[7]: https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_C.html?utm_source=chatgpt.com "The OpenCL™ C Specification - Khronos Registry"


---

# USER

I have to clarify:
1) I know the bottlenec is reall kernel execution overhed, and no data transfer. I know kernel executaion overhead is rather small in C/C++ but with pyOpenCL it is totally different stroyy, python is simply 100x or 500x slower than C++ and this is major bottlencek for Molecular dynamics.
2) I do not wat to use C/C++ compilation, wthat would completely break de architectorue of whole repo which has currenly onpy pyton and pyOpenCL depnecnecies and I want to keep it that way.
3) I do not know how it that with that kernel object creation,. Iw we can avoid reperated kernel creation it would be great
4) add recorded command buffer - I'm a bit skeptical here, I suspect we already tried it once and realized that creation of this buffer in python is still slow, and in the end it is not much fater (if at all). But it is tru that key is to eliminate other python overheads which are done every MD iteration

---

# Chat GPT 5.6 sol

Yes—your clarifications change the priority substantially. The immediate question is not how to redesign MD around workgroups, but:

> How much of the apparent “kernel-launch overhead” is actually avoidable PyOpenC([documen.tician.de][1])argument marshalling?

## 1. PyOpenCL may currently be using the slowest invocation pattern

This pattern is problematic:

```python
for istep in range(nstep):
    program.eval_forces(queue, global_size, local_size, args...)
```

Every evaluation of `program.eval_forces` creates a **new `pyopencl.Kernel` object**. PyOpenCL explicitly warns that caches important for efficient kernel enqueue are attached to the `Kernel` instance and are therefore ineffective in this usage pattern. ([documen.tician.de][1])this:

```python
program.eval_forces(...)
program.eval_forces(...)
program.eval_forces(...)
```

is not equivalent to reusing the same compiled kernel handle. The OpenCL program remains compiled, but PyOpenCL repeatedly performs something conceptually like:

```text
Python attribute lookup
    ↓
create cl_kernel object
    ↓
create Python Kernel wrapper
    ↓
construct/find argument invoker
    ↓
set every argument
    ↓
enqueue
    ↓
destroy/release temporary wrapper
```

For a kernel executing only a few microseconds, that can absolutely dominate.

The first essential distinction is therefore:

```python
# Bad for repeated fine-grained calls:
program.eval_forces(...)

# Persistent kernel object:
eval_forces = program.eval_forces
eval_forces(...)
```

PyOpenCL caches generated “invokers”—small Python snippets optimized for setting arguments and enqueuing—on persistent kernel objects. ([documen.tician.de][1])2. Even the persistent `kernel(...)` interface still sets all arguments every time

This is the next hidden cost.

When calling:

```python
eval_forces(
    queue,
    global_size,
    local_size,
    positions,
    forces,
    parameters,
    np.int32(natoms),
)
```

PyOpenCL's `Kernel.__call__()` does approximately:

```python
eval_forces.set_args(...)
cl.enqueue_nd_range_kernel(...)
```

on every invocation. This is documented explicitly. ([documen.tician.de][1])t means every MD step may involve:

* constructing the Python argument tuple;
* processing every argument;
* converting scalar arguments;
* calling `clSetKernelArg` for every argument;
* then calling `clEnqueueNDRangeKernel`.

For MD, this is often unnecessary because almost all kernel arguments are persistent:

* positions buffer;
* velocities buffer;
* forces buffer;
* neighbor-list buffer;
* atom parameters;
* simulation parameters;
* number of atoms;
* timestep;
* work sizes.

The **contents** of the buffers change, but their OpenCL buffer handles do not.

The lowest-overhead standard PyOpenCL path is therefore conceptually:

```python
eval_forces = program.eval_forces

eval_forces.set_args(
    positions,
    forces,
    parameters,
    simulation_state,
)

for istep in range(nstep):
    cl.enqueue_nd_range_kernel(
        queue,
        eval_forces,
        global_size,
        local_size,
    )
```

PyOpenCL documents this “stateful” interface directly: call `set_args()` once and then invoke `enqueue_nd_range_kernel()` on the retained kernel object. ([documen.tician.de][1])s does **not** solve the one-Python-call-per-kernel problem, but it eliminates a potentially large part of the Python-side overhead.

## 3. One persistent kernel object per argument configuration

Because OpenCL kernel arguments are stored in the kernel object, alternating argument layouts can be handled by retaining several kernel objects.

For ping-pong buffers:

```text
integrate_A_to_B:
    input  = state_A
    output = state_B

integrate_B_to_A:
    input  = state_B
    output = state_A
```

you can create two persistent kernel objects:

```python
integrate_ab = program.integrate
integrate_ba = program.integrate

integrate_ab.set_args(state_a, state_b, ...)
integrate_ba.set_args(state_b, state_a, ...)
```

Then the loop only enqueues the already-bound objects.

The same applies to variants such as:

* force kernel with or without neighbor rebuild;
* different integrator stages;
* constraint iteration kernels;
* odd/even timestep configurations.

The relevant abstraction is not merely “one kernel object per OpenCL kernel function,” but:

> one persistent kernel object per static argument binding.

## 4. Put dynamic scalar state in a device buffer

A remaining problem is scalar values that change each step:

```text
step index
simulation time
adaptive timestep
thermostat state
active/inactive flag
neighbor-list validity
```

Passing them as ordinary scalar kernel arguments requires another `set_arg()` call.

Instead, a small persistent device-side control structure can contain:

```c
typedef struct {
    int   istep;
    int   active;
    int   neighbor_valid;
    float dt;
    float time;
    float temperature;
} SimulationState;
```

All kernels receive one persistent pointer:

```c
__global SimulationState* sim;
```

and update or read its contents on the GPU.

Then the host-side kernel arguments never change. This is especially important for command buffers, but it also reduces ordinary PyOpenCL overhead.

It does **not** imply host-device transfers every step. The structure stays on the device.

## 5. What remains after this optimization

After persistent kernels and pre-bound arguments, the inner loop is reduced roughly to:

```python
enqueue(queue, kernel_A, global_A, local_A)
enqueue(queue, kernel_B, global_B, local_B)
```

But each call still causes:

* one Python-to-PyOpenCL transition;
* Python processing of the work-size tuples;
* the C-level OpenCL enqueue;
* construction of a returned Python `Event` wrapper.

`enqueue_nd_range_kernel()` always returns a new PyOpenCL `Event`. ([documen.tician.de][1])n when you write:

```python
cl.enqueue_nd_range_kernel(...)
```

without assigning the return value, the event wrapper is still created and then released.

Therefore, persistent objects can remove:

[
T_{\rm kernel-object}
+
T_{\rm invoker-setup}
+
T_{\rm set-args},
]

but not:

[
T_{\rm Python-call}
+
T_{\rm event-wrapper}
+
T_{\rm clEnqueueNDRangeKernel}.
]

This is the minimum overhead available through the normal PyOpenCL enqueue API.

## 6. Your “Python is 100–500× slower” observation needs one refinement

It is completely plausible that the current PyOpenCL call path costs 100–500 times more than the execution of a tiny GPU kernel.

But there are two different comparisons:

[
\frac{T_{\rm PyOpenCL\ full\ call}}
{T_{\rm GPU\ kernel}}
]

versus

[
\frac{T_{\rm PyOpenCL\ optimized\ enqueue}}
{T_{\rm C\ clEnqueueNDRangeKernel}}.
]

The first ratio can easily be enormous for tiny kernels.

The second may be much smaller after:

* retaining the kernel object;
* pre-binding arguments;
* retaining work-size tuples;
* avoiding explicit events;
* avoiding `wait_for`;
* avoiding `finish()` inside the loop.

PyOpenCL's low-level implementation is already native code; the expensive part may not be arbitrary Python bytecode execution but repeated wrapper/object/argument machinery triggered by the convenient call interface. ([documen.tician.de][2])before concluding that a new execution architecture is necessary, we should distinguish these three cases experimentally:

```text
A: program.kernel(...) every iteration
B: retained_kernel(...) every iteration
C: retained_kernel.set_args(...) once
   + enqueue_nd_range_kernel(...) every iteration
```

I would expect potentially a very large difference between A and C, although the exact factor must be measured on your system.

## 7. Command buffers: your skepticism is justified, but creation time is not the decisive quantity

A command buffer has two phases:

```text
record once:
    force
    integrate
    force
    integrate
    ... 100 times

replay many times:
    enqueue command buffer
    enqueue command buffer
    enqueue command buffer
```

The OpenCL extension is explicitly designed to separate command construction from enqueue and permit the same recorded sequence to be repeatedly submitted. ([Khronos Registry][3])refore, command-buffer creation being slow is acceptable only when:

[
N_{\rm replay},T_{\rm saved}

>

T_{\rm record}.
]

For example, a trajectory of (10^5) steps could use a 100-step command buffer replayed 1000 times:

[
\underbrace{T_{\rm record}(100)}*{\text{once}}
+
1000,T*{\rm replay}.
]

If the command buffer is reconstructed for every 100-step block, then it probably provides little or no benefit:

[
1000,T_{\rm record}(100)
]

simply replaces the original Python enqueue loop with a Python recording loop.

So the correct use would be:

```text
initialization:
    create one 100-step command buffer

simulation:
    replay it repeatedly
```

not:

```text
for every 100 MD steps:
    construct command buffer
    execute it once
    destroy it
```

## 8. Static MD buffers are actually well suited to recording

The fact that positions and velocities change does not require command-buffer reconstruction.

The recorded kernel argument is:

```text
cl_mem positions
```

not the current contents of `positions`.

The same recorded sequence can operate on:

```text
positions at t=0
positions at t=100
positions at t=200
...
```

because it always refers to the same device allocation.

A command buffer only needs rebuilding when something structural changes, such as:

* a different buffer object;
* different global or local dimensions;
* different kernel;
* different immutable scalar arguments;
* a different command sequence.

Consequently, for a fixed-capacity MD system, the command buffer could remain valid even if:

* particle positions change;
* forces change;
* neighbor-list contents change;
* simulation time changes;
* thermostat variables change.

Those are buffer contents, not command structure.

## 9. But current PyOpenCL does not expose command buffers

This is an important correction to treating command buffers as an immediately available PyOpenCL feature.

As of the current PyOpenCL repository, support for `cl_khr_command_buffer` is still an open enhancement request. ([GitHub][4])with the strict dependency constraint:

```text
Python
NumPy
PyOpenCL
```

there is currently no normal high-level PyOpenCL command-buffer API.

A possible architecture-preserving route would be a small **pure-Python `ctypes` wrapper** around the OpenCL extension functions:

```text
clCreateCommandBufferKHR
clCommandNDRangeKernelKHR
clFinalizeCommandBufferKHR
clEnqueueCommandBufferKHR
```

This would require no compiled C/C++ code and `ctypes` is part of Python's standard library. But it would be lower-level, driver-dependent, and only possible when the installed OpenCL implementation advertises `cl_khr_command_buffer`.

That is quite different from introducing a compiled native extension into your repository, but it is still additional low-level OpenCL binding code.

## 10. Why your previous command-buffer experiment may have shown no benefit

Several explanations are plausible:

1. **The command buffer was reconstructed for each execution.**
   Then recording overhead replaces ordinary enqueue overhead instead of being amortized.

2. **Arguments were still rebound every block.**
   That retains much of the Python work.

3. **The experiment was not using an actual KHR command buffer.**
   A Python list of events or queued commands does not collapse them into one OpenCL submission.

4. **The driver replay implementation did not optimize much.**
   Command buffers remove repeated host-side command construction, but the GPU may still execute the same individual kernel dispatches.

5. **The dominant cost was already device dispatch latency.**
   Command buffers do not eliminate the required global kernel boundaries.

6. **The test replayed the buffer only once.**
   Then the one-time recording cost can hide the saved submission cost.

7. **There was a synchronization or profiling operation inside each batch.**
   That can dominate both ordinary and recorded execution.

## 11. Revised ranking under your constraints

Given:

* no C/C++ source or compilation;
* retain Python + PyOpenCL architecture;
* no data transfer in the MD loop;
* exact global synchronization between kernels;

the possibilities should be ranked as follows.

### First: eliminate accidental PyOpenCL overhead

Use persistent kernel objects and never perform:

```python
program.kernel_name(...)
```

inside the MD loop.

### Second: pre-bind all kernel arguments

Prefer:

```python
kernel.set_args(...)
cl.enqueue_nd_range_kernel(...)
```

over:

```python
kernel(..., all_arguments...)
```

when the arguments are static.

### Third: represent changing parameters as device state

This prevents repeated `set_arg()` operations and makes the execution graph structurally static.

### Fourth: measure the irreducible ordinary-PyOpenCL enqueue cost

At this point the loop contains only bare enqueue calls. This tells us whether the remaining problem is truly the Python/PyOpenCL crossing.

### Fifth: investigate reusable command buffers

Only if:

* the driver supports `cl_khr_command_buffer`;
* the command buffer is recorded once and replayed many times;
* a pure-Python `ctypes` binding is acceptable.

The most important immediate finding is therefore that **repeated kernel-object creation and repeated `set_args()` are separate from the unavoidable kernel enqueue**. Your present performance may be measuring all three together:

[
T_{\rm measured}
================

T_{\rm create\ Kernel}
+
T_{\rm set\ args}
+
T_{\rm enqueue}
+
T_{\rm GPU\ dispatch}.
]

Before considering persistent global-barrier kernels or changing the MD decomposition, we need to isolate:

[
T_{\rm enqueue}
+
T_{\rm GPU\ dispatch}.
]

That optimized baseline may be dramatically better than the present PyOpenCL loop.

[1]: https://documen.tician.de/pyopencl/runtime_program.html "OpenCL Runtime: Programs and Kernels - pyopencl 2026.1.2 documentation"
[2]: https://documen.tician.de/pyopencl/?utm_source=chatgpt.com "pyopencl 2026.1.2 documentation"
[3]: https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/cl_khr_command_buffer.html "cl_khr_command_buffer(3)"
[4]: https://github.com/inducer/pyopencl/issues/820?utm_source=chatgpt.com "Support cl_khr_command_buffer #820 - inducer/pyopencl"
