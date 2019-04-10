from hangman import wordPicker, HangMan
import pdb

new_word = wordPicker()
hang = HangMan(new_word.word)
pdb.set_trace()
