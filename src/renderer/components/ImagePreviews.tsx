import React, { useState, useEffect } from 'react';
const { ipcRenderer } = require('electron');

import { Box, Button, Image, SimpleGrid, Flex } from '@chakra-ui/react';

const ImagePreviews = (filePaths: Array<string>) => {
    const [imagePreviews, setImagePreviews] = useState([]);
    const [currentPage, setCurrentPage] = useState(0);
    const imagesPerPage = 10;  // Change this as per your requirements

    const chooseUploadPath = async () => {
        if (filePaths.length > 0) {
            const previews = await ipcRenderer.invoke('get-previews', filePaths);
            setImagePreviews(previews);
        }
    };

    const totalPages = Math.ceil(imagePreviews.length / imagesPerPage);
    const startIndex = currentPage * imagesPerPage;
    const currentImages = imagePreviews.slice(startIndex, startIndex + imagesPerPage);

    return (
        <Box>
            <Button colorScheme="teal" onClick={chooseUploadPath}>Choose Folder</Button>
            <SimpleGrid columns={4} spacing={4}>
                {currentImages.map((preview, index) => (
                    <Box key={index} boxSize="128px">
                        <Image
                            src={`data:image/png;base64,${preview.data}`}
                            alt={'preview'}
                            boxSize="100%"
                            objectFit="cover"
                        />
                    </Box>
                ))}
            </SimpleGrid>
            <Flex justify="center" mt={4}>
                <Button
                    onClick={() => setCurrentPage(old => Math.max(old - 1, 0))}
                    disabled={currentPage === 0}
                    ml={2}
                    colorScheme="teal"
                >
                    Previous
                </Button>
                {Array.from(Array(totalPages), (_, i) => (
                    <Button key={i} onClick={() => setCurrentPage(i)} ml={2} colorScheme={i === currentPage ? "teal" : "gray"}>
                        {i + 1}
                    </Button>
                ))}
                <Button
                    onClick={() => setCurrentPage(old => Math.min(old + 1, totalPages - 1))}
                    disabled={currentPage === totalPages - 1}
                    ml={2}
                    colorScheme="teal"
                >
                    Next
                </Button>
            </Flex>
        </Box>
    );
};

export default ImagePreviews;