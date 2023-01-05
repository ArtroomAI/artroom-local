import React, { useState, useEffect } from 'react';
import { useRecoilState, useRecoilValue } from 'recoil';
import * as atom from '../atoms/atoms';
import ImageObject from './Reusable/ImageObject';
import {
    Box
} from '@chakra-ui/react';
import Masonry from 'react-masonry-css'
import { breakpoints } from '../constants/breakpoints';
import ImageModal from './Modals/ImageModal';

function ImageViewer () {
    var path = require('path');
    const image_save_path = useRecoilValue(atom.imageSavePathState);
    const batch_name = useRecoilValue(atom.batchNameState);
    const [imageViewPath, setImageViewPath] = useRecoilState(atom.imageViewPathState);
    const [imagePreviews, setImagePreviews] = useState(["",""]);

    const [showImageModal, setShowImageModal] = useRecoilState(atom.showImageModalState);

    useEffect(() => {
        if(imageViewPath.length <= 1){
            setImageViewPath(path.join(image_save_path,batch_name));
        }

    },[]);

    useEffect(() => {
        if (imageViewPath.length > 1) {
            window.getImages(imageViewPath).then((result) => {
                let imagesData = [];

                result.forEach((resultPath) => {
                  const imagePath = path.join(imageViewPath, resultPath);
                  imagesData.push(new Promise(async (resolve, reject) => {
                    window.getImageFromPath(imagePath).then((output) => {
                      resolve({imagePath, b64: output.b64, metadata: output.metadata});
                    });
                  }));
                });

                Promise.all(imagesData).then((results)=>{
                    console.log(results);
                    setImagePreviews(results);
                })    
            })
        }
    },[imageViewPath]);

    return (
        <>
        {showImageModal && <ImageModal/>}
        <Box 
            height="90%"
            ml="50px"
            p={5}
            rounded="md"
            width="75%" >
            <Masonry
                breakpointCols={breakpoints}
                className="my-masonry-grid"
                columnClassName="my-masonry-grid_column"
            >
            {imagePreviews.map((image, index) => (
                <Box py={2} px={1} key={index}>
                    <ImageObject b64={image.b64} metadata={image.metadata}/>
                </Box>
            ))}
            </Masonry>
        </Box>
        </>
    );
}

export default ImageViewer;
