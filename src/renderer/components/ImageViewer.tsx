import React, { useState, useEffect } from 'react';
import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import { Box, Icon, Button } from '@chakra-ui/react';
import Masonry from 'react-masonry-css'
import { breakpoints } from '../constants/breakpoints';
import ImageModal from './Modals/ImageModal/ImageModal';
import path from 'path';
import { batchNameState, imageSavePathState } from '../SettingsManager';
import { GoFileDirectory } from 'react-icons/go';

const ImageComponent = ({ image_path } : { image_path: string }) => {
    return <Box py={2} px={1}><img loading='lazy' src={image_path}/></Box>
}

const FolderComponent = ({ directory_path, directory_full_path } : { directory_path: string; directory_full_path: string }) => {
    const setImageViewPath = useSetRecoilState(atom.imageViewPathState);
    return <Box py={2} px={1}>
        { directory_path }
        <Icon
            as={GoFileDirectory}
            fontSize="xl"
            justifyContent="center"
            onClick={() => setImageViewPath(directory_full_path)} />
    </Box>
}

const PathButtons = () => {
    const [imageViewPath, setImageViewPath] = useRecoilState(atom.imageViewPathState);

    let absPath = '';

    const paths = imageViewPath.split(path.sep).map((el, index) => {
        absPath += index === 0 ? el : path.sep + el;
        return [el, absPath];
    });
  
    return <Box width="100%" pos="sticky" top="40px" bgColor='#000'>
        { paths.map(([relPath, absPath], index) => (
            <Button key={index} onClick={() => setImageViewPath(absPath)}>
                {relPath}
            </Button>
        )) }
    </Box>
};

function ImageViewer () {
    const imageSavePath = useRecoilValue(imageSavePathState);
    const batchName = useRecoilValue(batchNameState);
    const [imageViewPath, setImageViewPath] = useRecoilState(atom.imageViewPathState);
    const [imagePreviews, setImagePreviews] = useState<ImageViewerElementType[]>([]);

    const [showImageModal, setShowImageModal] = useRecoilState(atom.showImageModalState);

    useEffect(() => {
        if(imageViewPath === '') {
            setImageViewPath(path.join(imageSavePath, batchName));
        } else {
            window.api.imageViewer(imageViewPath, path.join(imageSavePath, batchName)).then((result) => {
                if(result.error) {
                    setImageViewPath(result.error.path);
                    return;
                }
                setImagePreviews(result.results);
            });
        }
    }, [batchName, imageSavePath, imageViewPath]);

    return (
        <>
            {showImageModal && <ImageModal/>}
            <Box
                height="90%"
                ml="50px"
                p={5}
                rounded="md"
                width="75%"
                pos="relative">

                <PathButtons />

                <Masonry
                    breakpointCols={breakpoints}
                    className="my-masonry-grid"
                    columnClassName="my-masonry-grid_column"
                >
                    {imagePreviews.map((image) => (
                        image.isFolder
                            ? <FolderComponent directory_path={image.name} directory_full_path={image.fullPath} />
                            : <ImageComponent image_path={image.fullPath} />
                    ))}
                </Masonry>
            </Box>
        </>
    );
}

export default ImageViewer;
