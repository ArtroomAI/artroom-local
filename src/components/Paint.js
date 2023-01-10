import 'tui-image-editor/dist/tui-image-editor.css';
import 'tui-color-picker/dist/tui-color-picker.css';

import React from 'react';
import { useState, useEffect, createRef } from 'react';
import { useInterval } from './Reusable/useInterval/useInterval';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import axios from 'axios';

import ImageEditor from '@toast-ui/react-image-editor';
import theme from '../themes/theme';
import { UnifiedCanvas } from './UnifiedCanvas/UnifiedCanvas';

import {
    Box,
    VStack,
    createStandaloneToast,
} from '@chakra-ui/react';
import { FaQuestionCircle } from 'react-icons/fa';
import Prompt from './Prompt';
function Paint () {
    useEffect(
        () => {
            const interval = setInterval(
                () => axios.get(
                    'http://127.0.0.1:5300/get_progress',
                    { headers: { 'Content-Type': 'application/json' } }
                ).then((result) => {
                    if (result.data.status === 'Success' && result.data.content.percentage) {
                        setProgress(result.data.content.percentage);
                        setRunning(result.data.content.running);
                        setStage(result.data.content.stage);
                    } else {
                        setProgress(-1);
                        setRunning(false);
                        setStage('');
                    }
                }),
                1500
            );
            return () => {
                clearInterval(interval);
            };
        },
        []
    );

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
