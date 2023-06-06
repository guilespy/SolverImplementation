import numpy as np
import cv2


def createVideo(originalImage,
                NoisyImage,
                model1,
                model2,
                name='outVisualTool.mp4',
                titel='Defect',
                CMAP = ((0,0,0),(255,255,255),(0,0,0),(0,0,0))
                ):
    """
    Cretates a video using the colors CMAP for the title
    of each Model (it should be adjusted depending on the pictures and
    experiment
    originalImage,NoisyImage : ndarray
    model1,model2 : should be instances of mySolvers
    """
    print("Creating video in " + name)
    
    n,m = np.shape(originalImage)
    coordinatesOriginal = (10,30)
    coordinatesNoisy = (m+10,30)
    coordinatesFRB = (10,n+30)
    coordinatesFBF = (10+m,30+n)


    pathFRB = model1.solutionsPath[:model1._bestIT]
    pathFBF = model2.solutionsPath[:model2._bestIT]
    
    nFRB = len(pathFRB)
    nFBF = len(pathFBF)

    if nFRB > nFBF:
        for i in range(nFRB - nFBF):
            pathFBF.append(pathFBF[-1])
    else:
        for i in range(nFBF-nFRB):
            pathFRB.append(pathFRB[-1])

    spFRB = np.asarray(pathFRB)
    spFBF = np.asarray(pathFBF)
    upLinks = []
    upRight = []
    for i in range(np.minimum(spFRB.shape[0],spFBF.shape[0])):
        upLinks.append(originalImage)
        upRight.append(NoisyImage)

    upLinks = np.asarray(upLinks)
    upRight = np.asarray(upRight)

    DATA_1 = np.concatenate((upLinks,upRight),axis=2)
    DATA_2 = np.concatenate((spFRB,spFBF),axis=2)
    DATA = np.concatenate((DATA_1,DATA_2),axis=1)
    DATA = 255*DATA
    DATA = np.uint8(DATA)

    size = DATA.shape[1], DATA.shape[2]
    fps = 25
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for _ in range(DATA.shape[0]):
        data = DATA[_]
        cv2.putText(data,
                    'Original',
                    coordinatesOriginal,
                    font,1,
                    (0,0,0),
                    2,
                    cv2.LINE_AA)
        cv2.putText(data,
                    titel,
                    coordinatesNoisy,
                    font,1,
                    CMAP[1],
                    2,
                    cv2.LINE_4)

        cv2.putText(data,
                    model1._Method,
                    coordinatesFRB,
                    font,1,
                    CMAP[2],
                    2,
                    cv2.LINE_4)
        cv2.putText(data,
                    model2._Method,
                    coordinatesFBF,
                    font,1,
                    CMAP[3],
                    2,
                    cv2.LINE_4)


        out.write(data)
    out.release()

def createPlot(*args,
               name='outVisualTool.pdf'):

    """
    creates a plot, the models come from mySolvers
    """

    from matplotlib import pyplot as plt
    print("Creating plot in " + name)
    for model in args:
        plt.plot(model.history,
                 label=model._Method)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('f(x)')
    plt.legend()
    plt.savefig(name)

