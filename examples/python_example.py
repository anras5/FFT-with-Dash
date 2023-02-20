print("Hello! What is your name?")
name = input()

if name == "Alice" or name == "Bob":
    print("Nice to meet you, " + name + "! You have a great name.")
else:
    print("Nice to meet you, " + name + "!")

num_letters = 0
for letter in name:
    if letter.isalpha():
        num_letters += 1

print("Your name has " + str(num_letters) + " letters.")
