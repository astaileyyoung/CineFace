from argparse import ArgumentParser

from analyze_video import analyze_video


def process_file(fp,
                 encoder='DeepID'):
    data = analyze_video(str(fp),
                         encode=True,
                         encoder=encoder) 
    


def main(args):
    """
    Pipeline:
    --Detect Faces
    --Represent Faces
    --Cluster Faces: Do I have to? If the vector search is fast enough, I can search every face in the movie or episode. In fact, this might limit errors since it removes a step. 
    --Store Faces
    --Identify Faces: Should I do this here or when I update the SQL server?

    Another thought: I could do everything from the SQL server script. The pipeline might make more sense located there since every step in the process would be done in one place. In fact, I might be able to do away with the CineFace repo altogether by moving the analysis scripts to the videotools repo, which might make more sense anyway. After thinking about it, the detect_faces function should be run on new video files, which is exactly what my SQL Server automation does. 

    Another way of dealing with the CineFace repo would be to keep it so that the SQL Server scripts dealing with faces would be better organized. Everything would be managed by SQL Server and CineFace just a container of scripts essentially.

    I think the second option may be better. The SQL Server repo will probably keep growing in size and complexity, so submodules may be necessary eventually anyway.

    The way the SQL Server watch.py works requires some thought. Currently, the script directly uploads the face file to GitHub, which is then pulled when it runs the server update script. 
    """
    pass 


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('')
    args = ap.parse_args()
    main(args)
