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
    Flex,
    Box,
    VStack,
    HStack,
    Button,
    FormControl,
    Tooltip,
    Stack,
    Text,
    Image,
    Progress,
    Select,
    createStandaloneToast,
    SimpleGrid
} from '@chakra-ui/react';
import { FaQuestionCircle } from 'react-icons/fa';
import Prompt from './Prompt';
function Paint () {
    const imageEditor = createRef();

    const { ToastContainer, toast } = createStandaloneToast();

    const [paintType, setPaintType] = useRecoilState(atom.paintTypeState);
    const [init_image, setInitImage] = useRecoilState(atom.initImageState);

    const [width, setWidth] = useRecoilState(atom.widthState);
    const [height, setHeight] = useRecoilState(atom.heightState);
    const [text_prompts, setTextPrompts] = useRecoilState(atom.textPromptsState);
    const [negative_prompts, setNegativePrompts] = useRecoilState(atom.negativePromptsState);
    const [batch_name, setBatchName] = useRecoilState(atom.batchNameState);
    const [steps, setSteps] = useRecoilState(atom.stepsState);
    const [aspect_ratio, setAspectRatio] = useRecoilState(atom.aspectRatioState);
    const [seed, setSeed] = useRecoilState(atom.seedState);
    const [use_random_seed, setUseRandomSeed] = useRecoilState(atom.useRandomSeedState);
    const [n_iter, setNIter] = useRecoilState(atom.nIterState);
    const [sampler, setSampler] = useRecoilState(atom.samplerState);
    const [cfg_scale, setCFGScale] = useRecoilState(atom.CFGScaleState);
    const [strength, setStrength] = useRecoilState(atom.strengthState);

    const [ckpt, setCkpt] = useRecoilState(atom.ckptState);
    const [image_save_path, setImageSavePath] = useRecoilState(atom.imageSavePathState);
    const [long_save_path, setLongSavePath] = useRecoilState(atom.longSavePathState);
    const [highres_fix, setHighresFix] = useRecoilState(atom.highresFixState);
    const [speed, setSpeed] = useRecoilState(atom.speedState);
    const [use_full_precision, setUseFullPrecision] = useRecoilState(atom.useFullPrecisionState);
    const [use_cpu, setUseCPU] = useRecoilState(atom.useCPUState);
    const [save_grid, setSaveGrid] = useRecoilState(atom.saveGridState);
    const [debug_mode, setDebugMode] = useRecoilState(atom.debugMode);
    const [ckpt_dir, setCkptDir] = useRecoilState(atom.ckptDirState);
    const [delay, setDelay] = useRecoilState(atom.delayState);

    const [progress, setProgress] = useState(-1);
    const [stage, setStage] = useState('');
    const [running, setRunning] = useState(false);
    const [open_pictures, setOpenPictures] = useRecoilState(atom.openPicturesState);
    const [keep_warm, setKeepWarm] = useRecoilState(atom.keepWarmState);

    const [mainImage, setMainImage] = useRecoilState(atom.mainImageState);
    const [latestImages, setLatestImages] = useRecoilState(atom.latestImageState);
    const [latestImagesID, setLatestImagesID] = useRecoilState(atom.latestImagesIDState);

    const [paintHistory, setPaintHistory] = useRecoilState(atom.paintHistoryState);

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
