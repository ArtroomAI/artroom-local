import { React, useState, useEffect } from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import ImageObj from './ImageObj';
import {
    Flex,
    createStandaloneToast
} from '@chakra-ui/react';
function ImageViewer () {
    const { ToastContainer, toast } = createStandaloneToast();
    const [navSize, changeNavSize] = useRecoilState(atom.navSizeState);
    const [image_save_path, setImageSavePath] = useRecoilState(atom.imageSavePathState);
    const [batch_name, setBatchName] = useRecoilState(atom.batchNameState);
    const [imageViewPath, setImageViewPath] = useRecoilState(atom.imageViewPathState);
    const [imagePaths, setImagePaths] = useState([]);
    const [imagePreviews, setImagePreviews] = useState([]);

    useEffect(
        () => {
        // If(imageViewPath.length === 0){
            setImageViewPath(`${image_save_path}/${batch_name}`);
        // }
        },
        []
    );

    useEffect(
        () => {
            if (imageViewPath.length > 0) {
                window.getImages(imageViewPath).then((result) => {
                    setImagePaths(result);
                    const images = [];
                    result.forEach((path) => {
                        window.getImageFromPath(path).then((output) => {
                            images.push(output);
                        });
                    });
                    setImagePreviews(images);
                });
            }
        },
        [imageViewPath]
    );

    return (
        <Flex
            align="center"
            justify="center"
            ml={navSize === 'large'
                ? '80px'
                : '0px'}
            transition="all .25s ease"
            width="100%">
            {
                imagePreviews?.map((image, index) => {
                    <ImageObj
                        B64={image.B64}
                        active={false}
                        imagePath={image.ImagePath}
                        key={index} />;
                })
            }
        </Flex>
    );
}

export default ImageViewer;
