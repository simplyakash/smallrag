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


# 🧠 SOLID Principles in Python

SOLID is a set of 5 Object-Oriented Design principles that help create:

```text
✅ Maintainable Code
✅ Extensible Code
✅ Testable Code
✅ Loosely Coupled Systems
```

Introduced by:

```text
Robert C. Martin ("Uncle Bob")
```

---

# S → Single Responsibility Principle (SRP)

## Definition

A class should have only one reason to change.

In other words:

```text
One Class
One Responsibility
```

### ❌ Bad Example

```python
class User:

    def save_to_db(self):
        pass

    def send_email(self):
        pass
```

### Why Bad?

```text
User Class is responsible for:
1. User Data
2. Database Operations
3. Email Operations
```

Too many responsibilities.

---

### ✅ Good Example

```python
class User:
    pass


class UserRepository:

    def save(self, user):
        pass


class EmailService:

    def send(self, user):
        pass
```

Responsibilities:

```text
User           → Data
UserRepository → Database
EmailService   → Email
```

---

# O → Open Closed Principle (OCP)

## Definition

Software entities should be:

```text
Open for Extension
Closed for Modification
```

Meaning:

```text
Add New Features
Without Changing Existing Code
```

### ❌ Bad Example

```python
class PaymentProcessor:

    def pay(self, method):

        if method == "card":
            pass

        elif method == "upi":
            pass
```

Adding:

```text
PayPal
Apple Pay
```

requires modifying existing code.

---

### ✅ Good Example

```python
from abc import ABC, abstractmethod


class PaymentMethod(ABC):

    @abstractmethod
    def pay(self):
        pass


class CardPayment(PaymentMethod):

    def pay(self):
        print("Card Payment")


class UPIPayment(PaymentMethod):

    def pay(self):
        print("UPI Payment")
```

New payment method:

```python
class PaypalPayment(PaymentMethod):

    def pay(self):
        print("Paypal Payment")
```

No existing code changes needed.

---

# L → Liskov Substitution Principle (LSP)

## Definition

Child classes should be replaceable for parent classes.
> A subclass should be able to replace its parent class without breaking the correctness of the program.

In simple words:

```text
If B is a subclass of A,

then anywhere A is used,
B should work correctly too.
```


### ❌ Bad Example

```python
class Bird:

    def fly(self):
        pass


class Penguin(Bird):

    def fly(self):
        raise Exception("Penguins cannot fly")
```

Problem:

```python
bird = Penguin()
bird.fly()
```

Breaks expectations.

---

### ✅ Good Example

```python
class Bird:
    pass


class FlyingBird(Bird):

    def fly(self):
        pass


class Sparrow(FlyingBird):
    pass


class Penguin(Bird):
    pass
```

Now substitution is safe.

---

# I → Interface Segregation Principle (ISP)

## Definition

Do not force classes to implement methods they do not need.

### ❌ Bad Example

```python
class Worker:

    def work(self):
        pass

    def eat(self):
        pass
```

Robot:

```python
class Robot(Worker):

    def eat(self):
        raise Exception()
```

Robot does not eat.

---

### ✅ Good Example

```python
class Workable:

    def work(self):
        pass


class Eatable:

    def eat(self):
        pass
```

Human:

```python
class Human(Workable, Eatable):
    pass
```

Robot:

```python
class Robot(Workable):
    pass
```

---

# D → Dependency Inversion Principle (DIP)

## Definition

Depend on abstractions.

Do not depend on concrete implementations.

---
# 🧠 Another Example of Dependency Inversion Principle (DIP)

## Definition

```text
High-level modules should not depend on low-level modules.

Both should depend on abstractions.
```

---

# ❌ Bad Example

Suppose we have a notification system.

---

## Email Service

```python
class EmailService:

    def send(self, message):
        print(f"Sending Email: {message}")
```

---

## Notification Manager

```python
class NotificationManager:

    def __init__(self):
        self.email_service = EmailService()

    def notify(self, message):
        self.email_service.send(message)
```

---

## Problem

```text
NotificationManager
        ↓
Depends Directly On
        ↓
EmailService
```

If tomorrow we want:

```text
SMS
WhatsApp
Slack
Teams
```

we must modify:

```python
NotificationManager
```

This violates DIP.

---

# ✅ Good Example (Using Abstraction)

## Step 1: Create Interface

```python
from abc import ABC, abstractmethod


class NotificationService(ABC):

    @abstractmethod
    def send(self, message):
        pass
```

---

## Step 2: Implement Email

```python
class EmailService(NotificationService):

    def send(self, message):
        print(f"Email: {message}")
```

---

## Step 3: Implement SMS

```python
class SMSService(NotificationService):

    def send(self, message):
        print(f"SMS: {message}")
```

---

## Step 4: High-Level Module Depends on Interface

```python
class NotificationManager:

    def __init__(self, service: NotificationService):
        self.service = service

    def notify(self, message):
        self.service.send(message)
```

---

## Usage

### Email

```python
email_service = EmailService()

manager = NotificationManager(email_service)

manager.notify("Order Placed")
```

---

### SMS

```python
sms_service = SMSService()

manager = NotificationManager(sms_service)

manager.notify("Order Placed")
```

No code changes required in:

```python
NotificationManager
```

---

# Real Production Example

## ❌ Bad

```python
class LLMApplication:

    def __init__(self):
        self.llm = OpenAI()
```

Problem:

```text
Tightly coupled to OpenAI
```

Switching to:

```text
Gemini
Claude
Llama
```

requires code changes.

---

# ✅ Good

## Abstract Interface

```python
class LLM(ABC):

    @abstractmethod
    def generate(self, prompt):
        pass
```

---

## OpenAI

```python
class OpenAILLM(LLM):

    def generate(self, prompt):
        return "OpenAI Response"
```

---

## Gemini

```python
class GeminiLLM(LLM):

    def generate(self, prompt):
        return "Gemini Response"
```

---

## Application

```python
class ChatApplication:

    def __init__(self, llm: LLM):
        self.llm = llm

    def chat(self, prompt):
        return self.llm.generate(prompt)
```

---

## Usage

```python
app = ChatApplication(OpenAILLM())
```

or

```python
app = ChatApplication(GeminiLLM())
```

No application code changes needed.

---

# Visual Diagram

## ❌ Without DIP

```text
ChatApplication
        ↓
    OpenAI
```

Tightly coupled.

---

## ✅ With DIP

```text
ChatApplication
        ↓
       LLM
      /   \
     /     \
 OpenAI   Gemini
```

Application depends on:

```text
LLM Interface
```

not on:

```text
OpenAI
Gemini
```

---

# Interview Answer

Dependency Inversion Principle states that high-level modules should depend on abstractions rather than concrete implementations. For example, a notification manager should depend on a NotificationService interface instead of directly depending on EmailService. This allows implementations such as Email, SMS, or WhatsApp to be swapped without changing the business logic, improving flexibility, testability, and maintainability.
---

# 📊 Summary Table
# 📊 SOLID Principles — Last Minute Interview Revision Sheet

| Principle | Full Form | Core Idea | Memory Trick | Bad Smell | Good Design Example | Interview One-Liner |
|------------|------------|------------|------------|------------|------------|------------|
| **S** | **Single Responsibility Principle (SRP)** | A class should have only one reason to change. | **One Class = One Job** | Class handles multiple responsibilities such as DB, Email, Validation, Logging, etc. | Separate User, UserRepository, EmailService | A class should have only one responsibility and therefore only one reason to change. |
| **O** | **Open Closed Principle (OCP)** | Software should be open for extension but closed for modification. | **Extend, Don't Modify** | Adding a new feature requires changing existing code with multiple if-else blocks. | Base interface + new subclasses for new functionality | New functionality should be added by extending existing code rather than modifying tested code. |
| **L** | **Liskov Substitution Principle (LSP)** | Child classes must be replaceable for parent classes without breaking behavior. | **Replace Safely** | Child class throws errors for methods inherited from parent. | Sparrow extends Bird, Penguin does not implement Flyable. | Derived classes should be substitutable for their base classes without altering correctness. |
| **I** | **Interface Segregation Principle (ISP)** | Clients should not be forced to depend on methods they do not use. | **Keep Interfaces Small** | Fat interfaces with many unrelated methods. | Separate Workable and Eatable interfaces. | Create small, focused interfaces rather than large generic ones. |
| **D** | **Dependency Inversion Principle (DIP)** | Depend on abstractions, not concrete implementations. | **Code to Interfaces** | Business logic directly creates EmailService, MySQL, OpenAI, etc. | Inject abstractions via constructor dependency injection. | High-level modules should depend on abstractions rather than low-level implementations. |

---

# 🧠 Quick Memory Tricks

| Principle | Memory Trick |
|------------|-------------|
| **S** | One Class = One Job |
| **O** | Extend, Don't Modify |
| **L** | Replace Child for Parent Safely |
| **I** | Small Interfaces Only |
| **D** | Code to Interfaces |

---

# 🚨 Common Interview Examples

| Principle | Bad Example | Good Example |
|------------|-------------|-------------|
| **SRP** | User class does DB + Email + Validation | Separate User, Repository, EmailService |
| **OCP** | Huge if-else payment processor | Payment interface + CardPayment, UPIPayment |
| **LSP** | Penguin inherits Bird.fly() | FlyingBird and NonFlyingBird hierarchy |
| **ISP** | Worker interface has work() and eat() | Separate Workable and Eatable |
| **DIP** | ChatApp directly uses OpenAI | ChatApp depends on LLM interface |

---

# 🤖 ML / AI Examples

| Principle | ML/AI Example |
|------------|-------------|
| **SRP** | Separate Retrieval, Reranking, Prompting, Generation modules |
| **OCP** | Add Gemini/Claude/OpenAI support without changing pipeline code |
| **LSP** | Any Vector Store implementation should behave like a VectorStore interface |
| **ISP** | Separate EmbeddingProvider and LLMProvider interfaces |
| **DIP** | RAG Pipeline depends on Retriever interface, not Pinecone or FAISS directly |

---

# 🎯 30-Second Interview Answer

```text
S → One Class, One Responsibility

O → Extend Existing Code, Don't Modify It

L → Child Objects Must Replace Parent Objects Safely

I → Create Small Focused Interfaces

D → Depend on Abstractions Instead of Concrete Implementations
```

---

# 🎤 10-Second Interview Answer

```text
SOLID principles are five object-oriented design principles that improve maintainability, scalability, flexibility, and testability of software systems.

S → Single Responsibility
O → Open Closed
L → Liskov Substitution
I → Interface Segregation
D → Dependency Inversion
```

---

# 🔥 Ultimate Memory Sentence

```text
One Job,
Extend Don't Modify,
Replace Safely,
Keep Interfaces Small,
Depend on Abstractions.
```

Remember this single sentence and you can reconstruct all five SOLID principles during an interview.

---

# 🎤 Interview Answer

SOLID is a set of object-oriented design principles that improves maintainability, extensibility, testability, and loose coupling. The five principles are Single Responsibility, Open-Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion. These principles are widely used in production systems to build scalable and maintainable software.

# asyncio in Python

`asyncio` is Python's built-in library for writing:

```text
Asynchronous
Concurrent
Non-Blocking
```

programs.

It is mainly used for:

- API Calls
- Web Scraping
- Database Queries
- Network Applications
- Chat Servers
- I/O-Bound Tasks

---

# Why Do We Need asyncio?

Suppose you need to call 3 APIs.

## Normal (Synchronous) Code

```python
import time

def fetch():
    time.sleep(2)

fetch()
fetch()
fetch()
```

Execution:

```text
2s + 2s + 2s
=
6 seconds
```

Each task waits for the previous one to finish.

---

## With asyncio

```python
import asyncio

async def fetch():
    await asyncio.sleep(2)

async def main():
    await asyncio.gather(
        fetch(),
        fetch(),
        fetch()
    )

asyncio.run(main())
```

Execution:

```text
≈ 2 seconds
```

All tasks run concurrently while waiting.

---

# Core Idea

While one task is waiting for:

```text
Network
Database
File System
API Response
```

the CPU can execute another task.

Instead of:

```text
Task 1
   ↓ Wait

Task 2
   ↓ Wait

Task 3
```

You get:

```text
Task 1 ─┐
Task 2 ─┼─ Waiting Together
Task 3 ─┘
```

---

# Important Keywords

## async

Defines an asynchronous function.

```python
async def fetch():
    pass
```

Meaning:

```text
This function can pause and resume execution.
```

---

## await

Pauses the current coroutine until an operation completes.

```python
await asyncio.sleep(2)
```

Meaning:

```text
I'm waiting.
Run some other task meanwhile.
```

---

# Example

```python
import asyncio

async def task(name):
    print(f"Start {name}")
    await asyncio.sleep(2)
    print(f"End {name}")

async def main():
    await asyncio.gather(
        task("A"),
        task("B"),
        task("C")
    )

asyncio.run(main())
```

Output:

```text
Start A
Start B
Start C

(2 seconds later)

End A
End B
End C
```

Total time:

```text
≈ 2 seconds
```

instead of:

```text
≈ 6 seconds
```

---

# Event Loop

The heart of asyncio.

```text
Event Loop
     ↓
Schedules Tasks
     ↓
Switches Tasks While Waiting
```

Visualization:

```text
Task A Waiting
      ↓
Run Task B
      ↓
Task B Waiting
      ↓
Run Task C
      ↓
Task A Ready
      ↓
Resume Task A
```

---

# What is a Coroutine?

An async function returns a:

```text
Coroutine
```

Example:

```python
async def hello():
    return "Hi"
```

Calling:

```python
hello()
```

returns:

```text
Coroutine Object
```

NOT:

```text
"Hi"
```

To execute it:

```python
await hello()
```

or

```python
asyncio.run(hello())
```

---

# asyncio vs Threads

## Threading

```text
Multiple OS Threads
```

### Advantages

```text
Easy to use with blocking code
```

### Disadvantages

```text
Higher memory usage
Context-switching overhead
```

---

## asyncio

```text
Single Thread
Single Event Loop
Multiple Coroutines
```

### Advantages

```text
Lightweight
Scalable
Handles thousands of connections
```

### Disadvantages

```text
Not suitable for CPU-intensive tasks
```

---

# When NOT to Use asyncio

Avoid asyncio for:

```text
Machine Learning Training
Image Processing
Video Encoding
Matrix Multiplication
Heavy Numerical Computation
```

These are:

```text
CPU-Bound Tasks
```

Use:

```text
Multiprocessing
```

instead.

---

# Real-World Use Cases

## Download Multiple URLs

```python
import aiohttp
import asyncio

async def fetch(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

---

## Async Database Access

Libraries:

```text
asyncpg
aiomysql
motor (MongoDB)
```

---

## High-Performance APIs

Frameworks using asyncio:

- FastAPI
- Sanic
- aiohttp

---

# Interview Answer

`asyncio` is Python's built-in asynchronous programming framework. It uses an event loop, coroutines (`async` and `await`), and non-blocking I/O to execute multiple I/O-bound tasks concurrently within a single thread. It is commonly used for API calls, database operations, web servers, and network-intensive applications.

---

# Quick Memory Tricks

## Threading vs asyncio

```text
Threading
   ↓
Multiple Threads

asyncio
   ↓
One Thread
Many Coroutines
```

---

## CPU-Bound vs I/O-Bound

```text
CPU-Bound
   ↓
Multiprocessing

I/O-Bound
   ↓
asyncio
```

---

## async vs await

```text
async
   ↓
Function Can Pause

await
   ↓
Pause Here
Run Something Else
```

# Multiprocessing vs Threading in Python

This is one of the most common Python interview questions.

The short answer:

```text
CPU-Bound Tasks
      ↓
Multiprocessing

I/O-Bound Tasks
      ↓
Threading
```

---

# Why?

Python has something called:

```text
GIL (Global Interpreter Lock)
```

The GIL allows only:

```text
One thread
```

to execute Python bytecode at a time.

Because of this:

```text
Multiple threads
≠
Multiple CPU cores
```

for CPU-heavy work.

---

# Threading

Threading creates:

```text
Multiple Threads
Inside One Process
```

Example:

```text
Process
├── Thread 1
├── Thread 2
├── Thread 3
└── Thread 4
```

All threads:

```text
Share Memory
```

---

## Best For

```text
Network Calls
API Requests
Database Queries
Reading Files
Downloading Files
Web Scraping
```

These are:

```text
I/O-Bound Tasks
```

because most time is spent:

```text
Waiting
```

not computing.

---

## Example

```python
from threading import Thread
import time

def task():
    time.sleep(2)
    print("Done")

threads = []

for _ in range(5):
    t = Thread(target=task)
    t.start()
    threads.append(t)

for t in threads:
    t.join()
```

Execution:

```text
≈ 2 seconds
```

instead of:

```text
≈ 10 seconds
```

---

# Multiprocessing

Multiprocessing creates:

```text
Multiple Processes
```

Each process has:

```text
Its Own Memory
Its Own Python Interpreter
Its Own GIL
```

Example:

```text
CPU Core 1 ← Process 1
CPU Core 2 ← Process 2
CPU Core 3 ← Process 3
CPU Core 4 ← Process 4
```

True parallelism.

---

## Best For

```text
Machine Learning
Image Processing
Video Encoding
Numerical Computation
Data Processing
Scientific Computing
```

These are:

```text
CPU-Bound Tasks
```

---

## Example

```python
from multiprocessing import Pool

def square(x):
    return x * x

if __name__ == "__main__":
    with Pool(4) as pool:
        result = pool.map(square, range(10))

    print(result)
```

Uses multiple CPU cores.

---

# CPU-Bound Example

Suppose:

```text
Calculate Prime Numbers
```

for:

```text
10 million numbers
```

Most time spent:

```text
Computing
```

not waiting.

Use:

```text
Multiprocessing
```

because threads are limited by the GIL.

---

# I/O-Bound Example

Suppose:

```text
Download 100 URLs
```

Most time spent:

```text
Waiting for Network
```

Use:

```text
Threading
```

or

```text
asyncio
```

---

# Memory Usage

## Threading

```text
Low Memory Usage
```

Threads share memory.

Example:

```text
1 Process
5 Threads
```

Memory:

```text
Shared
```

---

## Multiprocessing

```text
Higher Memory Usage
```

Each process has its own memory.

Example:

```text
5 Processes
```

Memory:

```text
Separate Copies
```

---

# Communication

## Threading

Easy communication.

```python
shared_list.append(...)
```

because memory is shared.

---

## Multiprocessing

Need:

```python
Queue
Pipe
Manager
Value
Array
```

because memory is isolated.

---

# Performance

## CPU-Bound

```text
Threading       ❌
Multiprocessing ✅
```

Reason:

```text
GIL
```

---

## I/O-Bound

```text
Threading       ✅
Multiprocessing ❌
```

Reason:

```text
Most time spent waiting.
```

---

# Interview Comparison Table

| Feature | Threading | Multiprocessing |
|----------|-----------|----------------|
| Unit | Thread | Process |
| Memory | Shared | Separate |
| GIL Affected | Yes | No |
| Parallel CPU Execution | No | Yes |
| Memory Usage | Low | High |
| Communication | Easy | Harder |
| Best For | I/O-Bound | CPU-Bound |

---

# Decision Tree

```text
Task Type?
     │
     ├── Waiting for Network/File/DB?
     │          ↓
     │      Threading
     │
     └── Heavy Computation?
                ↓
         Multiprocessing
```

---

# Real-World Examples

## Use Threading

```text
Web Scraper
Chat Server
API Gateway
Database Queries
File Upload Service
```

---

## Use Multiprocessing

```text
Training ML Models
Image Recognition
Video Processing
ETL Pipelines
Scientific Simulations
```

---

# Interview Answer

Threading is best for I/O-bound tasks such as API calls, database queries, and file operations because threads can make progress while waiting for I/O. Multiprocessing is best for CPU-bound tasks such as machine learning, image processing, and numerical computations because each process has its own Python interpreter and GIL, allowing true parallel execution across multiple CPU cores.
