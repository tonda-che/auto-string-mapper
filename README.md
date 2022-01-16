# auto-string-mapper
This mapper can take two lists of strings and propose a mapping based on how similar the strings are. For every string in one liste of string (the object passed to the parameter "from_column") we look for the most similar string in the other list of strings (the object passed to the parameter "to_column") in order to propose a mapping.

# installation
TBD

# example
Lets suppose we have a data set with movie categories like "Drama", "Action-Thriller", "Romance", "Horror", ... and then we have another database which has also other movie categories that look different, for example only with 3 characters "DRA", "ACT", "ROM", "HOR", "THR", ... and we want to map these two together into one consistent database.

```
from asm import AutoStringMapper
AutoStringMapper(
    from=["Drama", "Action-Thriller", "Romance", "Horror"],
    to=["DRA", "ACT", "ROM", "HOR", "THR"],
).get_mapping()
```

The result would be:
```
{"Drama": "DRA",
 "Action-Thriller": "ACT",
 "Romance": "ROM",
 "Horror": "HOR"}
 ```

Remark: Of course, string similarities never give you factual truth. Like in this example where you should still would need to decide whether and "Action-Thriller" should be mapped to "ACT" (as the mapper proposes) or "THR". Other strings might even be similar but factually completely different.

# applications
The auto-string-mapper facilitates two applications:
 - creating mappings between different data sets or data bases (data mapping)
 - mapping set of "dirty" user inputs into a set of clean and clear categories (data cleaning)

# documentation
TBD
