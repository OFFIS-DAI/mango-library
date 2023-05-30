import math
import random

print()
original_ethics_score = 1.0
new_ethics_score = 1.0
tier_size = 0.1
decay_rate = 0.01
i = 0
enough = True


def calc_ethics_score(enough, old_ethics_score):
    if enough == False:
        temp = math.floor(old_ethics_score * 10) / 10
        if (math.floor(float(temp)) + 1) > (float(temp) + 0.19):
            return float("{:.2f}".format(float(temp) + 0.19))
        else:
            return float("{:.2f}".format((math.floor(float(temp)) + 1)-decay_rate))
    else:
        lower_tier_end = (math.floor(old_ethics_score * 10) / 10)
        print(lower_tier_end)
        temp_ethics_score = float("{:.2f}".format(old_ethics_score - decay_rate))
        print(temp_ethics_score)
        if temp_ethics_score <= lower_tier_end:
            return lower_tier_end
        else:
            return temp_ethics_score


while i < 10:
    if random.randint(1, 10) > 5:
        enough = True
    else:
        enough = False
    new_ethics_score = calc_ethics_score(enough, new_ethics_score)
    print("enough was: " + str(enough))
    print("new ethics score " + str(new_ethics_score))
    i += 1
