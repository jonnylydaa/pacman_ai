"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    I introduced no noise to the grid.
    """

    answerDiscount = .9
    answerNoise = 0.0

    return answerDiscount, answerNoise

def question3a():
    """
    I had no noise, no living reward, and a lower answer discount. Low answer
    discount avoided the +10.
    """

    answerDiscount = 0.1
    answerNoise = 0.0
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    I intoduced a decent amount of noise and answer discount and then
    avoided the cliff by giving a negative living reward.
    """

    answerDiscount = 0.55
    answerNoise = 0.4
    answerLivingReward = -1.0

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    I introduced a little bit of noise and a big answer discount to
    get to the distant exit and risk cliffs.
    """

    answerDiscount = 0.7
    answerNoise = 0.1
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    I added a little bit of a negative living reward to avoid the cliffs, added
    some noise and a high answer discount to take the long route to get to
    the distant exit.
    """

    answerDiscount = 0.85
    answerNoise = 0.3
    answerLivingReward = -0.15

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    I simply gave no discount to getting closer to the answer (or exit) and
    gave a living reward in order to stay alive but not go towards a cliff
    or an exit.
    """

    answerDiscount = 0.0
    answerNoise = 0.0
    answerLivingReward = 2.0

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    There are no epsilon or learning rate that would learn the optimal policy
    in just 50 iterations. There needs to be many more iterations in order to
    accurately explore the optimal policy.
    """
    # answerEpsilon = 0.3
    # answerLearningRate = 0.5
    # return answerEpsilon, answerLearningRate
    return NOT_POSSIBLE

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
