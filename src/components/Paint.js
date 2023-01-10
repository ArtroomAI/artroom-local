import React from 'react';
import { UnifiedCanvas } from './UnifiedCanvas/UnifiedCanvas';
import {
    Box,
    VStack,
} from '@chakra-ui/react';
import Prompt from './Prompt';
function Paint () {
    return (
        <Box
            align="center"
            width="100%">
            <VStack
                align="center"
                spacing={4}>
                <Box
                    className="paint-output">
                    <UnifiedCanvas></UnifiedCanvas>
                </Box>
                <Box width="80%">
                    <Prompt />
                </Box>
            </VStack>
        </Box>
    );
}
export default Paint;
