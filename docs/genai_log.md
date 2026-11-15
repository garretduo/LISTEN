## Lyndon Huang

Tool for all rows: Claude (Anthropic), Opus 5.5, in the Claude app.

| Task | How checked | Where it appears |
|---|---|---|
| Drafted a team gameplan from the README and rubric | Team adjusted roles and dates to our own split | Planning only, not in the repo |
| Drafted the 50 lay-vs-legal minimal pairs | I read every pair, checked the everyday version has no legal terms and both versions tell the same facts, and edited where needed | `analysis/minimal_pairs.csv` |
| Helped with checking code for the app and for the model training scripts, clarified when I ran into issues. Also helped with the website design.
| Checked the minimal-pair results | Claude noticed the 30-step MLP gave the same top guess for every story; Garret fixed the MLP and I re-ran everything | `results/minimal_pairs_*.csv` |
| Helped with explanations for labeling causes for the error-labeling sheet. I read each story and if I couldn't decide on what the cause was I consulted claude to help me. 
| Help with git, Windows setup and debugging | Commands checked by running them | Not in the repo |

**Effect on results:** my shared labels (items 3–30) were AI-assisted, so the agreement
score partly reflects agreement with Claude's suggestions. The paper should say so next
to the kappa.

**Data handling:** my label sheet, which contains story text from the dataset, was
uploaded to Claude to get these suggestions. The team agreed AI help was allowed for
labeling.