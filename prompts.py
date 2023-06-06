DELIMITER = "####"  # for OpenAI's GPT3.5 model, this is 1 token

SYSTEM_PROMPT = f"""You will reply to queries about the texts provided below. The texts are notes taken by a computer vision engineer working at a robotics company, documenting his day-to-day work. The notes are about various topics, such as discussions with colleagues, meeting minutes, and random ideas, mostly related to computer vision topics.  The text is in Markdown format. Each note is separated by "{delimiter}". The start of a note is indicated by a date, in the following format: `# dd-mm-yyy`. Your goal is to provide references to relevant notes, based on the user's query, briefly explaining why the note is relevant. If you cannot find a good match, respond with the following reply: "No good matches were found to your query". Do not use information that is not in the notes. If you found relevant notes which match the query, reply in the following format: <date>: <short explanation of why the note is relevant> 

Here are the notes:""" + """ 

{context}

Query: {query}
Relevant notes:"""


NOTE_EXAMPLES = f"""{delimiter}
# 05-06-2023
John mentioned to me a paper titled "Two-view Geometry Scoring Without Correspondence", at this link: https://arxiv.org/pdf/2306.01596.pdf.  From my understanding, it addresses the problem of finding a good estimation of the Fundamental Matrix for two camera views.  The Fundamental Matrix combines the internal parameters of both cameras and their relative position and orientation. John thinks it may be useful for a two ZED camera scenario like with harvesting. He thinks it could also be helpful in knowing if we lost calibration because of damage.
{delimiter}

{delimiter}
# 14-02-2023
Quote from 2017/2018 Lilian Weng blog: "Overall YOLOv3 performs better and faster than SSD, and worse than RetinaNet but 3.8x faster."
{delimiter}

{delimiter}
# 16-03-2023
## Talk with David about realsense2 
- David did not install the realsense2 package
- Get the package from here: https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy
- looking at this package it seems like aligned depth is *off* by default 
- solution: fork this package and set `enable_sync` and `align_depth` to true in the xml file in `launch/include`
{delimiter}"""

