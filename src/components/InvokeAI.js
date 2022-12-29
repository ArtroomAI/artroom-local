import React, { useState } from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import Prompt from './Prompt';
import UnifiedCanvasDisplay from './InvokeAI/UnifiedCanvas/UnifiedCanvasDisplay';
import {
    Box,
    Flex,
    Button,
    VStack,
    createStandaloneToast
} from '@chakra-ui/react';

function InvokeAI () {
    const { ToastContainer, toast } = createStandaloneToast();
    const [navSize, changeNavSize] = useRecoilState(atom.navSizeState);
    const [focused, setFocused] = useState(false);

    return (
        <Flex
            align="center"
            justify="center"
            ml={navSize === 'large'
                ? '180px'
                : '100px'}
            transition="all .25s ease"
            width="100%">
            <Box
                align="center"
                width="100%">
                <VStack
                    align="center"
                    spacing={4}>
                    <UnifiedCanvasDisplay />

                    <Button
                        className="run-button"
                        ml={2}
                        width="250px">
                        'Run'
                    </Button>

                    <Box width="80%">
                        <Prompt setFocused={setFocused} />
                    </Box>
                </VStack>
            </Box>
        </Flex>
    );
}
export default InvokeAI;
