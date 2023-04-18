import React, { useState, useEffect } from 'react';
import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import {
    Box,
    Icon,
    Button,
    Flex,
    IconButton,
    NumberInput,
    NumberInputField,
    Text,
    Tooltip,
    Select,
    ButtonGroup,
    Input
} from '@chakra-ui/react';
import Masonry from 'react-masonry-css'
import { breakpoints } from '../constants/breakpoints';
import ImageModal from './Modals/ImageModal';
import path from 'path';
import { batchNameState, imageSavePathState } from '../SettingsManager';
import { GoFileDirectory } from 'react-icons/go';
import { BiArrowToLeft, BiArrowToRight, BiChevronLeft, BiChevronRight } from 'react-icons/bi';

const ImageComponent = ({ image_path, onClick } : { image_path: string, onClick: () => void }) => {
    return <Box onClick={onClick} py={2} px={1}><img loading='lazy' src={image_path}/></Box>
}

const FolderComponent = ({ directory_path, directory_full_path } : { directory_path: string; directory_full_path: string }) => {
    const setImageViewPath = useSetRecoilState(atom.imageViewPathState);
    return <Flex py={2} px={1} onClick={() => setImageViewPath(directory_full_path)}
        justifyContent="center" flexDir="column" alignItems="center">
        <Icon
            as={GoFileDirectory}
            fontSize="9xl"
            justifyContent="center"
            display="block" />
        <Box overflowWrap="break-word">{ directory_path }</Box>
    </Flex>
}

const SelectDisks = () => {
    const [imageViewPath, setImageViewPath] = useRecoilState(atom.imageViewPathState);
    const [disks, setDisks] = useState(['C:']);

    useEffect(() => {
        window.api.getDisks().then(setDisks)
    }, []);

    return <Box w="10%">
        <Select onChange={(e) => setImageViewPath(e.target.value)} value={`${imageViewPath.split(path.sep)[0]}\\`}>
            { disks.map((disk) => {
                return <option key={disk} value={`${disk}\\`}>{disk}</option>
            }) }
        </Select>
    </Box>
}

const PathButtons = () => {
    const [imageViewPath, setImageViewPath] = useRecoilState(atom.imageViewPathState);

    let absPath = '';

    const paths = imageViewPath.split(path.sep).filter(Boolean).map((el) => {
        absPath += el + path.sep;
        return [el, absPath];
    });
    return (
        <Box m={4}>
            <Flex mb={4}>
                <SelectDisks />
                <Input ml={2} value={imageViewPath} onChange={e => { setImageViewPath(e.target.value) }} />
            </Flex>
            <ButtonGroup isAttached variant="outline">
                { paths.map(([relPath, absPath], index) => (
                    <Button key={index} onClick={() => setImageViewPath(absPath)}>
                        {relPath}
                    </Button>
                )) }
            </ButtonGroup>
        </Box>
    )
}

const Controls = ({ pageIndex, maxLength, gotoPage } : { pageIndex: number; maxLength: number; gotoPage: React.Dispatch<React.SetStateAction<number>>}) => {
    return <Box width="100%" pos="sticky" top="30px" p="2" bgColor="#080B16">
        <PathButtons />
    
        <Flex justifyContent="space-between" m={4} alignItems="center">
            <Flex>
                <Tooltip label="First Page">
                    <IconButton
                        aria-label="arrowLeft"
                        onClick={() => gotoPage(0)}
                        isDisabled={pageIndex === 0}
                        icon={<BiArrowToLeft />}
                        mr={4}
                    />
                </Tooltip>
                <Tooltip label="Previous Page">
                    <IconButton
                        aria-label="chevronLeft"
                        onClick={() => gotoPage((e) => e - 1)}
                        isDisabled={pageIndex === 0}
                        icon={<BiChevronLeft />}
                    />
                </Tooltip>
            </Flex>

            <Flex alignItems="center">
                <Text flexShrink="0" mr={8}>
                    Page{" "}
                    <Text fontWeight="bold" as="span">
                        {pageIndex + 1}
                    </Text>{" "}
                    of{" "}
                    <Text fontWeight="bold" as="span">
                        {Math.ceil(maxLength / 100)}
                    </Text>
                </Text>
                <Text flexShrink="0">Go to page:</Text>{" "}
                <NumberInput
                    ml={2}
                    mr={8}
                    w={28}
                    min={1}
                    max={maxLength / 100}
                    onChange={(s) => {
                        const v = parseInt(s);
                        const page = v ? v - 1 : 0;
                        gotoPage(page);
                    }}
                    defaultValue={pageIndex + 1}
                >
                    <NumberInputField />
                </NumberInput>
            </Flex>

            <Flex>
                <Tooltip label="Next Page">
                    <IconButton
                        aria-label="chevronRight"
                        onClick={() => gotoPage(e => e + 1)}
                        isDisabled={maxLength < 100 * (pageIndex + 1)}
                        icon={<BiChevronRight />}
                    />
                </Tooltip>
                <Tooltip label="Last Page">
                    <IconButton
                        aria-label="arrowRight"
                        onClick={() => gotoPage(maxLength / 100 - 1)}
                        isDisabled={maxLength < 100 * (pageIndex + 1)}
                        icon={<BiArrowToRight />}
                        ml={4}
                    />
                </Tooltip>
            </Flex>
        </Flex>
    </Box>
};

function ImageViewer () {
    const imageSavePath = useRecoilValue(imageSavePathState);
    const batchName = useRecoilValue(batchNameState);
    const [imageViewPath, setImageViewPath] = useRecoilState(atom.imageViewPathState);
    const [imagePreviews, setImagePreviews] = useState<ImageViewerElementType[]>([]);
    const [imagePath, imageSetPath] = useState("");
    const setImageModalB64 = useSetRecoilState(atom.imageModalB64State);
    const setImageModalMetadata = useSetRecoilState(atom.imageModalMetadataState);
    const setShowImageModal = useSetRecoilState(atom.showImageModalState);
    const [pageIndex, gotoPage] = useState(0);

    const setImage = (image_path: string) => () => {
        window.api.getImageFromPath(image_path).then(result => {
            setImageModalB64(result.b64);
            setImageModalMetadata(JSON.parse(result.metadata ?? "{}"));
            imageSetPath(image_path);
            setShowImageModal(true);
        })
    }

    useEffect(() => {
        window.api.imageViewerChange((_, result) => {
            if(result.error) {
                setImageViewPath(result.error.path);
                return;
            }
            setImagePreviews(result.results);
        })
    }, []);

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
            <ImageModal imagePath={imagePath} />
            <Box
                height="90%"
                p={5}
                width="100%"
                pos="relative">

                <Controls pageIndex={pageIndex} gotoPage={gotoPage} maxLength={imagePreviews.length} />

                <Masonry
                    breakpointCols={breakpoints}
                    className="my-masonry-grid"
                    columnClassName="my-masonry-grid_column"
                >
                    {imagePreviews.slice(pageIndex * 100, (pageIndex + 1) * 100).map((image, index) => (
                        image.isFolder
                            ? <FolderComponent directory_path={image.name} directory_full_path={image.fullPath} key={index} />
                            : <ImageComponent image_path={image.fullPath} key={index} onClick={setImage(image.fullPath)} />
                    ))}
                </Masonry>
            </Box>
        </>
    );
}

export default ImageViewer;
