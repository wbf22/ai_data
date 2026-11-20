# Most valuable data

- tool calls and outputs
- message history
- model name
- system prompt and tool definitions if possible
- outcome evidence rather than subjective ratings
- task language, domain, difficulty, repo size
- specific rubrics for how the tasks are rated
- quality control, such as a second rater
- many ai providers restrict using outputs to train competing models
- no api keys, or personal information
- should be distinctive, ideally niche domain, difficult, long sessions, or very reliable labels
- should be 1000s of examples, diverse, high quality, well documented
- reproducible, not just my opinion
- you might rerate your work randomly weeks later to see how consistant you are. A good number there can make it more valuable
- keep good and bad data so that buyers can compare and understand the quality of the data
- the repo it was working in
    + any licenses or sub licenses in the repo
    + could possibly just include relevant files instead of whole repo

| Feature                  | Why it matters                                     |
| ------------------------ | -------------------------------------------------- |
| Real development work    | Less artificial than benchmark tasks               |
| Complete agent histories | Captures decisions, actions, and observations      |
| Human evaluations        | Provides quality signals that automated tests miss |
| Reproducible outcomes    | Allows researchers to independently verify results |


# Rubric Questions
- Did the agent succeed?
    + actually works
    + didn't break things
    + honest final message
    + no mistakes
- Did the agent follow the prompt?
    + nothing skipped
    + didn't go beyond
    + no major detours

Each of these should have a description if marked.


# Possible Monetary Value

| Scenario                     | Collection     | Hypothetical revenue |
| ---------------------------- | -------------- | -------------------: |
| Experiment                   | 50 sessions    | $0–$500              |
| Small collection             | 200 sessions   | $500–$2,000          |
| Established niche collection | 1,000 sessions | $2,000–$10,000       |
| Specialized collection       | 5,000 sessions | $10,000–$50,000      |


== Outcome ==
What was the outcome?
  1. succeeded - task fully accomplished
  2. partial - some of the task done, some not
  3. failed - task not accomplished
  select [1-3]: 1

  provide proof (e.g. file, command, test name, tool call)