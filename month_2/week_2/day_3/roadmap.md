# Day 9 Roadmap — Shannon Entropy (Week 2 Day 3, global Day 9)

1. **Watch** the two videos in `learn.md` — StatQuest first (the working definition),
   then 3Blue1Brown's Wordle video (why the formula has the shape it does).
2. **Math on paper:** surprise `-log₂(p)` at the extremes; fair coin (= 1 bit exactly);
   biased coin `P(heads) = 0.99`; fair die (= `log₂(6)`); the uniform-maximum argument
   in your own words. Then do the four practice questions in `learn.md`.
3. **Code problem 1:** implement `entropy(pmf)` in `code/entropy.py` until its tests pass.
4. **Code problem 2:** implement `entropy_of_text(text, level)` in `code/text_entropy.py`
   until its tests pass.
5. **Experiments:** fill in the `__main__` blocks — repetitive vs varied vs random text,
   char level vs word level — and write the compression comment (Problem 3).
6. Run the full suite: `pytest -v` from `month_2/week_2/day_3/`.
7. Run `/done` — tests must pass before the quiz.
