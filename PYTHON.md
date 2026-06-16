# 🐍 Python Interview Notes for ML / AI Engineers

---

# 1️⃣ Generators

## What is a Generator?

A generator is a function that produces values lazily using the `yield` keyword.

Instead of storing all values in memory, it generates values one at a time.

---

## Why Use Generators?

```text
✅ Memory Efficient
✅ Useful for Large Datasets
✅ Lazy Evaluation
```

---

## Example

```python
def count_up_to(n):
    i = 1

    while i <= n:
        yield i
        i += 1

g = count_up_to(5)

print(next(g))
print(next(g))
print(next(g))
```

Output:

```text
1
2
3
```

---

## Generator vs List

### List

```python
nums = [x*x for x in range(1000000)]
```

Memory:

```text
Stores all 1 million values
```

---

### Generator

```python
nums = (x*x for x in range(1000000))
```

Memory:

```text
Generates values on demand
```

---

## Interview Answer

A generator is an iterator-producing function that uses the `yield` keyword to return values lazily. It is memory-efficient because values are generated only when requested rather than stored in memory all at once.

---

# 2️⃣ Iterators

## What is an Iterator?

An iterator is an object that supports:

```python
__iter__()
__next__()
```

---

## Example

```python
nums = [1, 2, 3]

it = iter(nums)

print(next(it))
print(next(it))
print(next(it))
```

Output:

```text
1
2
3
```

---

## Iterable vs Iterator

### Iterable

```python
list
tuple
dict
set
string
```

Supports:

```python
__iter__()
```

---

### Iterator

Supports:

```python
__next__()
```

---

## Custom Iterator

```python
class Counter:

    def __init__(self, max_val):
        self.max_val = max_val
        self.current = 1

    def __iter__(self):
        return self

    def __next__(self):

        if self.current > self.max_val:
            raise StopIteration

        value = self.current
        self.current += 1

        return value
```

---

## Interview Answer

An iterable is any object that can return an iterator. An iterator is an object that maintains state and produces the next value using `__next__()` until `StopIteration` is raised.

---

# 3️⃣ Decorators

## What is a Decorator?

A decorator modifies or extends the behavior of a function without changing its source code.

---

## Example

```python
def logger(func):

    def wrapper():

        print("Starting function")

        result = func()

        print("Finished function")

        return result

    return wrapper


@logger
def hello():
    print("Hello")


hello()
```

Output:

```text
Starting function
Hello
Finished function
```

---

## Equivalent

```python
hello = logger(hello)
```

---

## Common Uses

```text
Logging
Authentication
Caching
Monitoring
Retry Logic
```

---

## Interview Answer

Decorators are higher-order functions that take another function as input and return a modified function. They are commonly used for logging, authorization, caching, and instrumentation.

---

# 4️⃣ GIL (Global Interpreter Lock)

## What is GIL?

GIL is a lock that allows only one thread to execute Python bytecode at a time.

---

## Why Does It Exist?

```text
Memory Safety
Reference Counting Simplicity
```

---

## CPU Bound Example

```python
def compute():
    for _ in range(100000000):
        pass
```

Threads:

```text
Do NOT scale well because of GIL
```

---

## IO Bound Example

```python
requests.get(url)
```

Threads:

```text
Work well because GIL is released during I/O
```

---

## Interview Answer

The Global Interpreter Lock ensures that only one thread executes Python bytecode at a time. It limits CPU-bound multithreading but has little impact on I/O-bound applications.

---

# 5️⃣ Asyncio

## What is Asyncio?

Asyncio enables cooperative multitasking using an event loop.

---

## Example

```python
import asyncio

async def fetch():

    print("Fetching")

    await asyncio.sleep(2)

    print("Done")


asyncio.run(fetch())
```

---

## Multiple Tasks

```python
async def task1():
    await asyncio.sleep(1)

async def task2():
    await asyncio.sleep(1)

asyncio.gather(
    task1(),
    task2()
)
```

---

## Best For

```text
API Calls
Database Calls
Web Scraping
Network Services
```

---

## Interview Answer

Asyncio provides concurrency through an event loop and coroutines. It is ideal for I/O-bound workloads because tasks can yield control while waiting for external operations.

---

# 6️⃣ Multiprocessing

## What is Multiprocessing?

Creates multiple processes instead of threads.

Each process has:

```text
Separate Memory
Separate Python Interpreter
Separate GIL
```

---

## Example

```python
from multiprocessing import Process

def worker():
    print("Running")


p = Process(target=worker)

p.start()
p.join()
```

---

## When To Use

```text
CPU Intensive Workloads
Model Training
Data Processing
Image Processing
```

---

## Interview Answer

Multiprocessing bypasses the GIL by creating separate processes. It is preferred for CPU-bound workloads because each process can execute in parallel on different CPU cores.

---

# 7️⃣ Memory Management

## Python Memory Management

Python uses:

```text
Reference Counting
+
Garbage Collection
```

---

## Reference Counting

```python
a = []

b = a
```

Reference count:

```text
2
```

---

## Object Deleted

```python
del a
```

Reference count:

```text
1
```

---

## Garbage Collector

Handles cyclic references.

Example:

```python
class A:
    pass

a = A()
b = A()

a.ref = b
b.ref = a
```

Reference counting alone cannot clean this.

GC handles it.

---

## Interview Answer

Python primarily uses reference counting for memory management and a cyclic garbage collector to reclaim objects involved in reference cycles.

---

# 8️⃣ Dictionary Internals

## How Does Dictionary Work?

Python dictionaries use:

```text
Hash Tables
```

---

## Example

```python
d = {
    "name": "Akash"
}
```

Lookup:

```python
d["name"]
```

Complexity:

```text
O(1)
```

Average case.

---

## Process

```text
Key
 ↓
Hash Function
 ↓
Bucket
 ↓
Value
```

---

## Complexity

| Operation | Complexity |
|------------|------------|
| Lookup | O(1) |
| Insert | O(1) |
| Delete | O(1) |
| Worst Case | O(n) |

---

## Interview Answer

Python dictionaries are implemented using hash tables. Keys are hashed to determine bucket locations, enabling average O(1) insertion, lookup, and deletion.

---

# 9️⃣ Time Complexity Cheat Sheet

## List

| Operation | Complexity |
|------------|------------|
| Append | O(1) |
| Access | O(1) |
| Search | O(n) |
| Insert Middle | O(n) |
| Delete Middle | O(n) |

---

## Dictionary

| Operation | Complexity |
|------------|------------|
| Lookup | O(1) |
| Insert | O(1) |
| Delete | O(1) |

---

## Set

| Operation | Complexity |
|------------|------------|
| Search | O(1) |
| Insert | O(1) |
| Delete | O(1) |

---

## Heap

| Operation | Complexity |
|------------|------------|
| Insert | O(log n) |
| Extract Min | O(log n) |
| Peek | O(1) |

---

# 🔟 OOP

## Four Pillars

```text
Encapsulation
Inheritance
Polymorphism
Abstraction
```

---

## Encapsulation

```python
class Employee:

    def __init__(self):
        self.__salary = 1000
```

---

## Inheritance

```python
class Animal:
    pass

class Dog(Animal):
    pass
```

---

## Polymorphism

```python
class Dog:
    def speak(self):
        return "Woof"

class Cat:
    def speak(self):
        return "Meow"
```

---

## Abstraction

```python
from abc import ABC, abstractmethod

class Shape(ABC):

    @abstractmethod
    def area(self):
        pass
```

---

## Interview Answer

OOP organizes software around objects and classes. The four pillars are encapsulation, inheritance, polymorphism, and abstraction, which improve modularity and maintainability.

---

# 1️⃣1️⃣ Context Managers

## What is a Context Manager?

Ensures setup and cleanup of resources.

---

## Example

```python
with open("file.txt") as f:
    data = f.read()
```

File automatically closes.

---

## Custom Context Manager

```python
class MyContext:

    def __enter__(self):
        print("Start")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Cleanup")
```

Usage:

```python
with MyContext():
    print("Inside")
```

---

## Interview Answer

Context managers manage resources safely using `__enter__()` and `__exit__()`. They ensure cleanup even when exceptions occur.

---

# 1️⃣2️⃣ Threading vs Asyncio

## Threading

```python
from threading import Thread
```

Uses OS threads.

Good for:

```text
I/O Bound Tasks
```

---

## Asyncio

```python
async def fetch():
    await api_call()
```

Uses:

```text
Single Thread
Event Loop
```

---

## Comparison

| Feature | Threading | Asyncio |
|-----------|-----------|----------|
| Uses OS Threads | Yes | No |
| Context Switch Cost | High | Low |
| Memory Usage | Higher | Lower |
| Best For | I/O Bound | Massive I/O Bound |
| CPU Parallelism | No (GIL) | No |

---

## Example

Threading:

```python
Thread(target=download)
```

Asyncio:

```python
await download()
```

---

## Interview Answer

Threading uses operating system threads and is suitable for moderate I/O-bound workloads. Asyncio uses a single-threaded event loop and scales efficiently to thousands of concurrent network operations with lower memory overhead.

---

# 🎤 Most Common Python Interview Questions

```text
1. What is a Generator?
2. Iterator vs Iterable?
3. What are Decorators?
4. Explain GIL.
5. Threading vs Multiprocessing?
6. What is Asyncio?
7. How does Python manage memory?
8. How are Dictionaries implemented?
9. Explain OOP pillars.
10. What is a Context Manager?
11. Threading vs Asyncio?
12. Why are Generators memory efficient?
```

For ML/AI Engineer interviews, these topics are among the highest-frequency Python questions.