This is an automatic method that works for Cell phenotype classification.
We integrate NIMBUS channel-wise scores with the biological cell-phenotype chart to classify the cell phenotype, rather than the traditional cluster method, which is unstable, requires human effort, and includes ambiguity.
This project is inspired by [NIMBUS][Nimbus] and is a post-processing from [Nimbus_repository][repo].

[Nimbus]: https://www.nature.com/articles/s41592-025-02826-9
[repo]: https://github.com/angelolab/Nimbus-Inference


# Requirements
If you need the AI to formulate the cell phenotype lineage, you need to get an API KEY from Claude platform [Claudeplate][plate];
Otherwise, you can skip this step.

[plate]: https://platform.claude.com/dashboard

# Installation
```bash
conda env create -f environment.yml
```
# Folder structure
Set the file arc as below:

`Agent`

`├── results` 

`├── Cellmarker2` 

`├── Chains` 

`├── data` 

`├── results` 

`├── environment.yml` 

`├── main.sh`

`├── scripts_.py`

# Running code

If you are using the method at first time, you will need to get the phenotype lineage tree, and save the structure into a JSON file from the tree's bottom to top. 

```bash
Whole_pipline.sh
```
If you have run the code before, or you want to manually correct the phenotye family tree, you can go to the `Chains` folder, modify it and use it directly, in this case, you don't need the agent to search the database, and generate the formatted json file.
```bash
Lineage_correction.sh
```
