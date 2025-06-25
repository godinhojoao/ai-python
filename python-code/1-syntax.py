# to run: python3 1-syntax.py

# 1. Printing single quotes or double (both work)
print("Hello")
print("Hello")
print("Hello", "joao")
print("Hello" + " joao")
print("hello5x" * 5)  # Python allows this (multiplying strings)
print(len("testing"))  # 7
print("hello".index("o"))  # 4
# print("hello".index("w")) # ValueError: substring not found
print("hello".count("l"))  # 2
print("hello".upper())  # HELLO
print("HELLO".lower())  # hello
print("hello".startswith("he"))  # True
print("hello".endswith("lo"))  # True
print("Hello world!".split(" "))  # ['Hello', 'world!']
print("hello", end=" ")  # end=" " avoid breaking line, default is \n
print("\n")

# slice of strings using start:stop:step (slice operator) -> it works for any lists or string
print("Hello world!"[3:6])  # lo (start: 3, stop: 6, step: 1)
print("Hello world!"[6:])  # world! (start: 6, stop: end, step: 1)
print(
    "Hello world!"[::-1]
)  # !dlrow olleH -> (start: init, stop: end, step: -1) it reverse the string since step is -1

x = None  # valid
print(x)  # prints: None
# print(y)       # NameError: name 'y' is not defined


# To receive input

# myInput = input("what's your name?")
# print(f"yourname is {myInput}")  # if needed we could try to convert like int(myInput)

print("---------------------")
# 2. Defining variables
mystring = "joao"
myint = 7
myfloat = 0.7

# print(mystring + myint + myfloat) # TypeError: can only concatenate str (not "int") to str
print(
    mystring, myint, myfloat
)  # joao 7 0.7 (it converts each argument to string automatically)
print(myint + myfloat)  # 7.7
print(str(myint) + str(myfloat) + mystring)  # 70.7joao
print(
    f"{mystring} {myint} {myfloat}"
)  # prints as a formatted string (f-string - formatted string literal)
print("%s and %d and %.2f" % (mystring, myint, myfloat))  # same formatting style as c


# There are no constants in python just the name convention: MY_CONST (but it's not protected)
# You can achieve immutability in other ways, for example: a class with @property without setter
class Constants:
    @property
    def MY_CONST(self):
        return 10

    @property
    def ANOTHER_CONST(self):
        return 20


c = Constants()
print(c.MY_CONST)  # 10
print(c.ANOTHER_CONST)  # 20
# c.ANOTHER_CONST=13 # AttributeError: property 'ANOTHER_CONST' of 'Constants' object has no setter

print("---------------------")

# 3. Lists -> behind scenes it's a dynamic array, but actually it depends of the interpreter that is running our python code.

# We can save different types on lists, since python is not a typed language.
myListA = []
myListA.append(1)
myListA.append("string")
myListA.append(0.7)

# print(myListA[10]) # IndexError: list index out of range
for value in myListA:
    print(value)

# "Concatenating" lists (joining):
myListB = []
myListB.append(30)
joinedLists = myListA + myListB
print("joinedLists", joinedLists)  # joinedLists [1, 'string', 0.7, 30]

print("test: ", joinedLists[0:1])  # test: test: [1]

# Just as in strings we can, multiply to repeat the sequence x times
print(
    "otherList:", joinedLists * 3
)  # otherList: [1, 'string', 0.7, 30, 1, 'string', 0.7, 30, 1, 'string', 0.7, 30]
print("---------------------")

# 4. Conditions

name = "Joao"
age = 4
if age == 4 and name == "Joao2":
    print("it went here")
elif name == "Joao":
    print("it went on elif")
else:
    print("it went on else")

if name in ["John", "Joao"]:
    print("your name is either john or joao")

# is check only type not value
if type(name) is str:
    print("type(name) is a string")

name2 = 2
if name is name2:
    print("name and name2 have same type")
else:
    print("name and name2 have different types")

if name != "john":
    print("your name isn't john")

name3 = None
if not name3:  # logical not
    print("your name isn't a truthy value")

print("---------------------")

# 5. Loops

for value in range(3, 8, 2):  # start: 3, stop: 8, step: 2 (skip 2 by 2)
    print(value, end=" ")  # 3, 5, 7
print("\n")

for index, value in enumerate(myListA):
    print(index, "-", value)

for value in myListA:
    if value == "string":
        break
    print(value)
# Output: 1

for value in myListA:
    if value == "string":
        continue
    print(value, end=" ")
    # Output: 1 0.7
print("\n")


count = 0
while count < 5:
    print(count, end="")  # 01234
    count += 1
print("\n")

print("---------------------")


# 6. Functions
def sayHello(username, age):
    print("Hello %s with age: %d" % (username, age))


sayHello("joao", 2)


# rest operator to have variable number of arguments (*rest or *list) -> rest or spread operator
def sumAllNumbers(*rest):
    total = 0
    for value in rest:
        total += value
    return total


print(sumAllNumbers(2, 3, 4, 5))  # 14

sumOfAllNumbersFrom1To100 = sumAllNumbers(
    *range(1, 101)  # * spread the list in multiple separated values
)
print(sumOfAllNumbersFrom1To100)  # 5050


# **options collects all rest arguments into a dictionary
def receiveObjectNamedVariables(**options):
    print(f"name: {options.get('name')} - age: {options.get('age')}")


receiveObjectNamedVariables(name="joao", age=4)

print("---------------------")


# 7. Classes and objects
class User:
    def __init__(self, name, age, country):  # constructor
        self.name = name
        self.__age = age  # "private" attribute,  can be accessed with joao._User__age
        self.__country = country

    def sayName(self):
        print(f"Inside: {self.name}")

    # only a getter to avoid changes
    @property
    def country(self):
        return self.__country


joao = User("joao", 2, "Brasil")
print(f"Outside: {joao.name}")
# print(joao.__age)  # AttributeError: 'User' object has no attribute '__age'

# works but it's a workaround to access "private" attributes
print(f"outside {joao._User__age}")

joao.sayName()
print("joao.country:", joao.country)
# joao.country = "test" # AttributeError: property 'country' of 'User' object has no setter


# inheritance (there are also abstract classes, methods, but I will not cover it here)
class Animal:
    def speak(self):
        print("Animal sound")


class Dog(Animal):
    def speak(self):
        print("Bark")


dog = Dog()
dog.speak()  # Bark

print("---------------------")

# 8. Dictionaries -> hashmaps same as objects in javascript (key-value stores with access by key O(1))
# Dictionaries are iterables
phonebook1 = {"John": 938477566, "Jack": 938377264, "Jill": 947662781}  # example1
phonebook = {}
phonebook["joao"] = 123
phonebook["john"] = 1234
print(phonebook)  # {'joao': 123, 'john': 1234}

# iterating over a dictionary

print(phonebook["joao"])  # 123
print(phonebook.get("joao"))  # 123
print(phonebook.values())  # dict_values([123, 1234]) -> iterable
print(phonebook.keys())  # dict_keys(['joao', 'john']) -> iterable
print(phonebook.items())  # dict_items([('joao', 123), ('john', 1234)]) -> iterable

for name, number in phonebook.items():
    print("name: %s - phone number: %d" % (name, number), end=" ")

print("\n")

for key in phonebook:
    print(f"key: {key}", end=" ")
print("\n")

del phonebook["joao"]  # deletes joao
print(phonebook.get("joao"))  # None
phonebook.pop("john")
print(phonebook.get("john"))  # None
print(phonebook)  # {}


print("---------------------")

# 9. Modules and Packages
# mathUtils is a package (folder), and prints.py is a module inside it.

# import * is discouraged since it pollutes the namespace introducing many variables at the same scope
# from mathUtils.prints import *  # (import all funcs from package)
from mathUtils.prints import printSum, printDiff  # specific import
from mathUtils.prints import printSum as printsumtest  # custom import name using "as"

printsumtest(5, 3)  # 8
printSum(2, 3)  # 5
print(printDiff(5, 2))  # 3

# To configure specifically what each package exports create a __init__.py
# then we could do like this without needing to tell from which package we want to import each func
from mathUtils import sum, diff

# without the __init__.py it would return error ImportError: cannot import name 'sum' from 'mathUtils' (unknown location)
print(sum(2, 3))  # 5

# Executing modules as scripts
# python3 scripts/fibo.py <arguments>
# python3 scripts/fibo.py 5

print("---------------------")

# 10. Exception Handling
# Exceptions are errors detected during execution (runtime errors).
# They occur after the code passes syntax checks but something goes wrong when running (e.g., dividing by zero, accessing a missing key).

# basic structure
# try:
#   code that may raise an error
# except SomeError:
#   handle that error
# finally:
#   always runs

# we could also handle multiple exceptions:
# except (TypeError, ValueError):

# we could also handle all exceptions (not recommended in most cases)
# except Exception as e:

list = [1, 2, 3, 4, 5]

for i in range(7):
    try:
        print(list[i], end=" ")
    except IndexError:  # Raised when accessing a non-existing index of a list
        print(0, end=" ")  # threat all others as 0s

print("\n")


try:
    raise Exception("spam", "eggs")  # raise can force an exception to occur
except Exception as inst:
    print(type(inst))  # the exception type
    print(inst.args)  # arguments stored in .args
    print(inst)  # __str__ allows args to be printed directly,
    # but may be overridden in exception subclasses
    x, y = inst.args  # unpack args
    print("x =", x)
    print("y =", y)

print("---------------------")

# 11. More Data Structures https://docs.python.org/3/tutorial/datastructures.html

# 11.1 Using lists as Stacks
stack = [3, 4, 5]
stack.append(6)  # [3, 4, 5, 6]
stack.pop()  # [3, 4, 5]
stack.pop()  # [3, 4]
stack.pop()  # [3]

## 11.2 Using lists as Queues
from collections import deque

queue = deque(["Joao"])
queue.append("John")  # deque(['Joao', 'John'])
queue.popleft()  # deque(['John']) FIFO
queue.popleft()  # # deque([])

## 11.3 List Comprehensions
# List comprehension is a short way to create a list using a single line of code. -> [expression for item in iterable if condition]
even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(f"even_squares: {even_squares}")  # even_squares: [0, 4, 16, 36, 64]

# Nested loop on list comprehension
pairs = [(x, y) for x in [1, 2] for y in [3, 4]]  # list of tuples [(x,y)]
print(f"pairs: {pairs}")  # pairs: [(1, 3), (1, 4), (2, 3), (2, 4)]

## 11.4 Tuples and Sequences
# A tuple is a sequence of values, like a list, but immutable.

t = (1, 2, "hello!")
print(t[0])  # 1
print(t)  # (1, 2, 'hello!')
# t[0] = 88888 # TypeError: 'tuple' object does not support item assignment

# tuples can be nested
u = (t, (1, 2, 3, 4, 5))
print(u)  # ((1, 2, 'hello!'), (1, 2, 3, 4, 5))

# Tuples can be "joined"
t1 = (1, 2)
t2 = (3, 4)
t3 = t1 + t2
print(t3)  # (1, 2, 3, 4)

# Tuples can contain mutable objects:
v = ([1, 2, 3], [3, 2, 1])
v[0].append(4)
print(v)  # ([1, 2, 3, 4], [3, 2, 1])

# Unpacking tuples
t = (1, 2, 3)
x, y, z = t
print("x,y,z:", x, y, z)
# Unpacking lists
t = [1, 2, 3]
x, y, z = t
print("x,y,z:", x, y, z)
# Unpacking dictionary keys
d = {"name": "Joao", "age": 2}
k1, k2 = d
print(k1, k2)  # name age

## 11.5 Sets (same as set's theory)
# An unordered collection with no duplicate elements.
# Allows set operations (union, intersection, etc.)

# Duplicates are removed automatically.
basket = {"apple", "orange", "apple", "pear", "banana"}
print(basket)  # {'orange', 'banana', 'pear', 'apple'}

print("orange" in basket)  # True
print("test" in basket)  # False

# Set operations
a = set(["test1", "test2"])
b = set(["test2"])
print(f"a - b: {a - b}")  # in a but not in b
print(f"a | b: {a | b}")  # union "or"
print(f"a & b: {a & b}")  # intersection "and"
print(f"a ^ b: {a ^ b}")  # symmetric difference (in a or b, but not both)

# Set comprehension (like list comprehension)
a = {x for x in "abracadabra" if x not in "abc"}
print("set compreheension", a)  # {'r', 'd'}

print("---------------------")

# 12. Async IO
# Used to write asynchronous, non-blocking code, typically to handle I/O bound tasks (e.g., file/network access, database with async drivers).
# We make it using coroutines https://docs.python.org/3/library/asyncio-task.html#coroutine
import asyncio

async def say_hello():
    await asyncio.sleep(1)
    print("Hello")

async def main():
    await say_hello()

asyncio.run(main())


# concurrent example
async def task(name, delay):
    await asyncio.sleep(delay)
    print(f"Task {name} finished after {delay} sec")

async def main():
    await asyncio.gather(
        task("A", 2),
        task("B", 1),
    )

asyncio.run(main()) # print B and then A

print("---------------------")
