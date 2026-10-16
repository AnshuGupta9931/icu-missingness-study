# Start here — what this project is, in plain language

You do not need to run anything to read this. Nothing here assumes you have
used Python before.

---

## 1. The problem, told as a story

A patient is admitted to intensive care. Over the first 24 hours the staff
record things about them: heart rate, blood pressure, oxygen level, and a set
of blood tests — lactate, creatinine, haemoglobin and so on.

Now here is the thing that makes this project interesting: **nobody gets every
test.** A blood test is ordered when a doctor has a reason to order it. A
monitor is attached when someone decides it is needed. So when you pull this
data out of the hospital system and lay it out as a table — one row per
patient, one column per measurement — the table has **holes in it**. Lots of
them. In our data, about 9% of all the boxes are empty, and for some tests like
lactate it is closer to 45%.

Now you want to build a computer model that predicts which patients will die,
so staff can prioritise. But almost every prediction model needs a **complete**
table. It cannot handle holes.

So you fill the holes in with guesses. That guessing has a technical name:
**imputation**. The simplest guess is "put the typical value for that column in
every hole." Cleverer methods look at the patient's other measurements and
guess something tailored to them.

**And now the actual research question of your project:**

> If you guess the missing values more accurately, do you actually get a better
> prediction of who dies?

It sounds like the answer must obviously be yes. Your project exists because
**it often isn't.** That is the whole point.

---

## 2. The one-sentence version

If someone asks what your project is about, say this:

> "When patient data has gaps, you have to fill them in before you can predict
> anything. I'm testing whether filling them in more accurately actually leads
> to better predictions of ICU death — and it turns out it often doesn't."

That is genuinely the whole thing. Everything else is detail.

---

## 3. Why the *reason* for the hole matters

This is the part that trips people up, so here it is slowly. Statisticians
split missing data into three kinds, based on **why** the value is absent.

**MCAR — Missing Completely At Random.**
The value is gone for a reason that has nothing to do with the patient. A
machine broke. A sample got dropped. A form was lost. The holes are scattered
randomly. This is the harmless case.

**MAR — Missing At Random.** (A terrible name — it does not mean random.)
Whether the value is missing depends on *something else you can see* about the
patient. Example: older patients in this ward happen to get fewer routine
bloods. You can *see* their age, so you can reason about the gap. Age explains
the hole; the hole is predictable from data you have.

**MNAR — Missing Not At Random.**
Whether the value is missing depends on **the missing value itself**. Example:
lactate is only measured when a doctor already suspects it is high. So if
lactate is missing, that very absence tells you it was probably normal. This is
the nasty case, because the information you would need to fix it is exactly the
information you do not have.

Why this matters for you: **these three situations need different handling, and
a method that works beautifully for one can fail for another.** Your project
measures that.

---

## 4. The trick at the heart of the project

Here is the problem with studying this on real data:

> To check whether your guess was right, you need to know the true value. But
> the true value is missing. That is why you were guessing.

You cannot grade an exam when you do not have the answer key.

**The solution — and this is the clever bit of your project — is to make your
own holes.**

Take values that *are* recorded. Hide them yourself, deliberately, following a
rule you chose. Now ask the computer to guess them. You can compare the guess
to the real value, because **you are the one who hid it.** You kept the answer
key.

And because you chose the hiding rule, you know whether you created an MCAR
situation, a MAR situation, or an MNAR situation. On real data you could never
know this. By simulating it, you do.

This is called **controlled masking**, and it is why your proposal says
(section 7) that the study "will not claim that observational data alone can
identify the true mechanism."

---

## 5. So what is the "synthetic cohort"?

Short answer: **fake patients that behave like real ones.**

Your project's real dataset is **MIMIC-IV** — a genuine database of intensive
care records from a Boston hospital. But it contains real people's medical
records, so you cannot just download it. You need to complete an ethics
training course and sign a data use agreement. **That is the access you have
asked your teacher for.** Until it arrives, you have no data.

So the code includes a generator that invents 8,000 fake ICU patients. Not
random noise — patients built to behave realistically:

- sicker patients have higher heart rates, lower blood pressure, higher lactate
- the risk of dying depends on those values in a realistic, non-simple way
  (for example, shock — high lactate *combined with* low blood pressure — is
  much worse than either alone)
- and crucially, the fake data has **realistic holes**: lactate is missing more
  often in the patients who were not very sick, exactly as in a real hospital

**Think of it as a flight simulator.** You would not learn to fly by getting
into a real plane first. The simulator lets you check that every control
responds correctly, that the instruments read sensibly, and that you know what
you are doing — before you touch the real aircraft.

**What this means for you:**

- The synthetic results are **not** findings about real patients. Do not put
  them in your dissertation as if they were.
- They **do** prove the machinery works end to end, which is genuinely
  valuable, and is exactly what your proposal calls a "pilot study" (section 12
  says "A pilot study will be run first to validate the pipeline").
- When MIMIC-IV access arrives, you change one line in a settings file and run
  the same command. The real results come out the other end.

---

## 6. What the results you saw actually say

The file `RESULTS.md` has numbers in it. Here is what they mean, and how much
to trust them.

**The main finding.** Across 84 different situations, the imputation method
that guessed missing values most accurately was *also* the method that
predicted death best only **59.5% of the time**. Under MNAR — the nasty case —
only **46.4%**, which is worse than flipping a coin.

In plain terms: **choosing your gap-filling method by how accurately it fills
gaps picks the wrong method about a third of the time.** That is the answer to
your proposal's fourth research question, and it is a real result rather than a
boring confirmation.

**A second finding worth knowing.** AUROC is the standard score people quote
for this kind of model. It dropped by only about 6% as the data got worse. But
AUPRC — a score that focuses on correctly finding the patients who actually
die, which is the thing you care about — dropped by **21%**. So the standard
score makes the damage look about four times smaller than it is.

**A third one.** KNN imputation (one of the four methods your proposal lists)
falls apart badly when a lot of data is missing. It ends up worse than just
filling every hole with the column's typical value.

**How much should you trust these?** They are from the simulator. Treat them as
"this is the shape of the analysis and these are the kinds of conclusions it
produces," not as facts about ICU patients. The real numbers arrive with
MIMIC-IV.

---

## 7. What is actually in this folder

```
START_HERE.md      <- you are reading this
README.md          the technical manual (read it second)
RESULTS.md         findings from the simulator run
CHANGELOG.md       a dated log of every change, which your proposal asks for

Project_102_revised.pdf    your original proposal

src/icumiss/       the actual code, in pieces:
    data/            makes the fake patients; reads MIMIC-IV when you have it
    missingness.py   punches the holes (MCAR / MAR / MNAR)
    imputation.py    the four gap-filling methods
    models.py        the four prediction models
    metrics.py       the scoring
    experiment.py    runs one combination start to finish
    runner.py        runs all 264 combinations

configs/           settings files - which experiment to run
    pilot.yaml           small and quick, for checking things work
    full_synthetic.yaml  the full experiment on fake data (already run)
    full_mimic.yaml      the full experiment on real data (for later)

scripts/           the commands you actually type
results/           everything that came out: tables and graphs
tests/             102 automatic checks that the code is correct
```

---

## 8. What you should actually do

### Right now, before access arrives

**You do not need to run anything.** The experiment has already been run once
on the fake data and the output is sitting in `results/full_synthetic/`.

What is genuinely worth your time:

1. **Read sections 1–4 above until you could explain them to a friend.** This
   is the part your teacher will test you on. The code is less important than
   understanding why the project is set up this way.
2. **Look at one picture.** Open
   `results/full_synthetic/figures/auroc_vs_level.png`. Left to right the three
   panels are the three kinds of missingness. Going right within each panel
   means more missing data. Going down means worse prediction. That one image
   is most of your results chapter.
3. **Start the literature review.** Your proposal (section 10) says
   implementation begins *after* the literature review and the novelty is
   settled. The code being ready early is a bonus, not a reason to skip that.

If you want to prove to yourself the thing runs, this takes about a minute:

```
cd /Users/anshugupta/Desktop/102-Project
.venv/bin/python scripts/run_experiments.py --config configs/pilot.yaml
```

Nothing is harmed by running it. It writes to `results/pilot/` and nowhere else.

### When your teacher gives you MIMIC-IV access

1. Download MIMIC-IV from PhysioNet. It is large — tens of gigabytes.
2. Open `configs/full_mimic.yaml` and change the line
   `root: /path/to/mimic-iv` so it points at wherever you saved it.
3. Run this first, and **read what it prints**:
   ```
   .venv/bin/python scripts/verify_mimic.py --root /path/to/mimic-iv
   ```
   It checks that the codes the project uses for "heart rate", "lactate" and so
   on actually mean those things in your copy of the database. These codes
   change between versions of MIMIC. If this step is wrong, everything
   downstream is confidently wrong, which is the worst kind of wrong.
4. Then run the real experiment:
   ```
   .venv/bin/python scripts/run_experiments.py --config configs/full_mimic.yaml
   ```
   It will take hours. It can be interrupted and restarted; it picks up where
   it stopped.

---

## 9. Questions your teacher might ask

**"Why didn't you just use the real missing values in the data?"**
Because you cannot score a guess when you do not know the true answer. By
hiding values yourself you keep the answer key. It also means you know which
mechanism you are testing, which you can never know from real data.

**"How do you know your MCAR / MAR / MNAR simulations are really different?"**
There is a check for exactly this. Run
`.venv/bin/python scripts/verify_design.py --config configs/pilot.yaml`.
It measures whether the hiding was linked to the hidden value, or to other
visible information, or to nothing. The three come out clearly distinct, and
the amount hidden matches what was requested to three decimal places.

**"How do you know your results aren't just a leak?"**
Data leakage means the model accidentally saw the answers during training.
Everything is split into training and test groups *before* any gaps are
punched, and the gap-filling methods only ever learn from the training group.
There is a test that corrupts the entire rest of the test set and confirms one
patient's filled-in values do not budge.

**"Why three repeats of everything?"**
Because random chance moves these numbers. Running each combination three times
with different random seeds shows how much is a real effect and how much is
noise. In 38 of the 84 situations, the gap between methods was *smaller* than
the random variation — meaning there was no real difference, and saying one
method won would have been wrong. The tables mark those cases.

**"Is this your own work?"**
Be straightforward: you specified the project, and the implementation was
written with AI assistance, which you reviewed. Say so if asked — most
institutions now have a disclosure policy, and the `CHANGELOG.md` documents
what was done and why. Check your university's rules.

---

## 10. Glossary

| Term | Plain meaning |
|---|---|
| **Cohort** | The group of patients in your study |
| **ICU mortality** | Whether the patient died during their intensive care stay. This is the thing being predicted |
| **Feature** | One column of the table — one measurement, like heart rate |
| **Imputation** | Filling in a missing value with a guess |
| **MCAR / MAR / MNAR** | The three reasons a value can be missing (section 3) |
| **Masking** | Deliberately hiding values you can actually see, so you can grade the guesses |
| **MIMIC-IV** | The real ICU database, needs permission to access |
| **Synthetic cohort** | Realistic fake patients, used while waiting for the real data |
| **Baseline** | The simplest method, here "fill every hole with the column's typical value." Everything else has to beat it to be worth using |
| **Median / KNN / MICE / MissForest** | The four gap-filling methods, roughly simplest to most complex |
| **Logistic regression / XGBoost / CatBoost / LightGBM** | The four prediction models. The first is simple and explainable; the other three are more powerful |
| **AUROC** | A 0–1 score for how well the model separates who died from who lived. 0.5 is guessing, 1.0 is perfect |
| **AUPRC** | Like AUROC but focused on correctly catching the patients who died. More honest when deaths are rare |
| **Brier score** | Whether the model's stated probabilities are trustworthy. Lower is better |
| **Seed** | A number fixing the random choices, so re-running gives identical results |
| **Leakage** | When a model accidentally sees test answers during training, making it look better than it is |
| **Pipeline** | The whole chain: load data → split → hide values → fill them → predict → score |
