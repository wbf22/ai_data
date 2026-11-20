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


Yeah that'd be great! This would be a script that would call a python program, and use a config file in the repo to determine what to zip up. This would be used to capture ai training data. This is what we need:


Start Command
- python script
  + looks for .ai_data_config file
  + reads what files and folders to include (.gitignore style)
  + zips up those files into src.zip
- opencode command to call that script

End Command
- call /export in opencode to get the session md file
- python script
  + copies the session md file from the repo too /home/brandon/Documents/ai_data/data/<session>/
  + copies the src.zip file from the repo too /home/brandon/Documents/ai_data/data/<session>/
  + runs an interactive data labeling for me to fill out:
    * Cli flow
        ```
        - Any problems? if yes:
            1. broken code
            2. skipped part of prompt
            3. went beyond prompt
            4. bad code or code style that works
            5. dishonest final message
            6. other
        - Anything done exceptionally well? if yes:
            1. clever solution
            2. first try
            3. well integrated in existing code
            4. other
        - Overall description
        ```
    * The script should ask those 3 questions requiring an answer, and only show the numbered questions if the user hits yes for one of them. Then any of the numbered questions can be chosen and a description is collected for each of those chosen questions
    * Then a full rubric is generated with both the main and numbered questions. If one of the main questions is answered 'no', then the numbered questions are all marked 'no'. 
  + then saves the data labeling rubric along with the src.zip and session md file in /home/brandon/Documents/ai_data/data/<session>/