import React, { useState } from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../../atoms/atoms';
import {
    Modal,
    ModalOverlay,
    ModalContent,
    ModalHeader,
    ModalBody,
    ModalFooter,
    Flex,
    Box,
    Text,
    Button,
    Icon
} from '@chakra-ui/react';
import {BiImages} from 'react-icons/bi'
import {GoSettings} from 'react-icons/go'
import {AiFillFolderOpen} from 'react-icons/ai'
import ImageModalObj from '../Reusable/ImageModalObj';

function ImageModal () {

    const [imageModalB64, setImageModalB64] = useRecoilState(atom.imageModalB64State);
    const [imageModalMetadata, setImageModalMetadata] = useRecoilState(atom.imageModalMetadataState);
    const [showImageModal, setShowImageModal] = useRecoilState(atom.showImageModalState);

    function handleClose() {
        setShowImageModal(false)
      }
    
    return (
        <Modal 
            size = '6xl'
            isOpen={showImageModal} 
            onClose={handleClose}
            scrollBehavior='outside'>
            <ModalOverlay />
            <ModalContent>
            <ModalHeader>
            </ModalHeader>
            <ModalBody>
                <Flex>
                <Box w="60%">
                    <Box>
                    <ImageModalObj b64={imageModalB64}></ImageModalObj>
                    <Button borderRadius="10" variant="outline" variantColor="blue" mr={2} fontSize="14"  fontWeight="normal">
                        <Icon mr={2} as={BiImages} />
                        Copy Image
                    </Button>
                    <Button borderRadius="10" variant="outline" variantColor="blue" mr={2} fontSize="14" fontWeight="normal" isDisabled={!('text_prompts' in imageModalMetadata)}
                    >
                        <Icon mr={2} as={GoSettings} />
                        Copy Settings
                    </Button>
                    <Button borderRadius="10" variant="outline" variantColor="blue" mr={2} fontSize="14" fontWeight="normal">
                        <Icon mr={2} as={BiImages} />
                        Set Starting Image
                    </Button>
                    <Button borderRadius="10" variant="outline" variantColor="blue" mr={2} fontSize="14" fontWeight="normal">
                        <Icon mr={2} as={AiFillFolderOpen} />
                        View in Files
                    </Button>
                    </Box>
                </Box>
                <Box w="40%" display="flex" flexDirection="column" alignItems="flex-start">                     
                <Box display="flex" alignItems="center" mt={2}>
                    <Text fontWeight="bold" color="blue">{imageModalMetadata.text_prompts}:</Text>
                </Box>
                <Text>{imageModalMetadata.text_prompts}</Text>
                <Box display="flex" alignItems="center" mt={2}>
                    <Text fontWeight="bold" color="blue">{imageModalMetadata.negative_prompts}:</Text>
                </Box>
                <Text>{imageModalMetadata.negative_prompts}</Text>
                <Box display="flex" alignItems="center" mt={2}>
                    <Text fontWeight="bold" color="blue">{imageModalMetadata.seed}:</Text>
                </Box>
                <Text>{imageModalMetadata.seed}</Text>
                <Box display="flex" alignItems="center" mt={2}>
                    <Text fontWeight="bold" color="blue">{imageModalMetadata.sampler}:</Text>
                </Box>
                <Text>{imageModalMetadata.sampler}</Text>
                </Box>

                </Flex>
            </ModalBody>
            <ModalFooter>
                <Button onClick={handleClose}>Cancel</Button>
            </ModalFooter>
            </ModalContent>
        </Modal>
    )
}

export default ImageModal;