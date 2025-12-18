This is an AI-agent that works for Cell phenotype classification.
We integrate NIMBUS channel-wise scores with the biological cell-phenotype chart to classify the cell phenotype, rather than the traditional unstable cluster method, which is unstable, requires human effort, and includes ambiguity.
This project is inspired by [NIMBUS][Nimbus] and is a post-processing from [Nimbus_repository][repo].

[Nimbus]: https://www.nature.com/articles/s41592-025-02826-9
[repo]: https://github.com/angelolab/Nimbus-Inference
# Requirements
You need to get an API KEY from claude plateform [Claudeplate][plate]
[plate]: https://platform.claude.com/dashboard

# Installation
```bash
conda env create -f environment.yml
```
