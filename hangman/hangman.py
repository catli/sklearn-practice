from urllib import urlopen
import re
import random as rand
import pytest

class wordPicker():

    def __init__(self):
        self.word = self.pickWord()

    def pickWord(self):
        word_reader = urlopen('http://norvig.com/ngrams/sowpods.txt')
        word_output = []
        for word in word_reader:
            word_output.append(str(word.splitlines()[0]))
        word = rand.sample(word_output, 1)
        return word[0].lower()



class HangMan:

    def __init__(self, word):
        self.word = word
        self.hidden = [True]*len(word)
        self.guessed = set()
        self.allowed_guesses = 6
        self.displayGame()


    def displayGame(self):
        # display the current state of the game 
        display = ''
        for i, char in enumerate(self.word):
            if self.hidden[i]:
                display = display + '_' + ' '
            else:
                display = display + char + ' '
        print(display)
        return display

    def guess(self, letter):
        if not re.match(r'([a-zA-Z])', letter):
            print('Not a letter!')
            return
        # make sure letter is not already guessed
        is_in_word = False
        if letter.lower() in self.guessed:
            print('Try again! Already guessed!')
            return
        else:
            for i, char in enumerate(self.word):
                if self.hidden[i]:
                    if char.lower() == letter.lower():
                        self.hidden[i] = False
                        is_in_word = True
            self.checkGameState()
            self.guessed.add(letter.lower())
            if not is_in_word:
                self.allowed_guesses-=1
                print('%s not in word!' % (letter))
                self.displayHang()
                self.checkGameOver()
        self.displayGame()


    def checkGameState(self):
        if not max(self.hidden):
            print('YOU WON!!')
            self.displayGame()



    def checkGameOver(self):
        if self.allowed_guesses==0:
            print('GAME OVER!!!!! HANGED!')
            self.hidden = [False]*len(self.word)
            self.displayGame()


    def displayHang(self):
        print('Your Hangman:')
        if self.allowed_guesses <= 5:
            print(' o ')
        if self.allowed_guesses <= 2:
            print('-|-')
        elif self.allowed_guesses ==3:
            print('-| ')
        elif self.allowed_guesses == 4:
            print(' | ')
        if self.allowed_guesses == 1:
            print('/ ')
        elif self.allowed_guesses == 0:
            print('/ \\')


def test_GuessLetter():
    test_word = 'testword'
    testhang = HangMan(test_word)
    testhang.guess('t')
    assert testhang.allowed_guesses == 6
    testhang.guess('a')
    # make sure that the hidden state is same length as test word
    assert len(testhang.hidden) == len(test_word)
    assert sum(testhang.hidden) == len(test_word)-2
    assert testhang.allowed_guesses == 5

def test_Display():
    test_word = 'testword'
    testhang = HangMan(test_word)
    assert testhang.displayGame() == '_ _ _ _ _ _ _ _ '
    testhang.guess('t')
    assert testhang.displayGame() == 't _ _ t _ _ _ _ '




