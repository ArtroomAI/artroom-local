import React from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import {
    Box,
    Flex,
    Tab,
    TabList,
    TabPanel,
    TabPanels,
    Tabs
} from '@chakra-ui/react';
import Upscale from './Upscale';

export default function ImageEditor () {
    const [navSize, changeNavSize] = useRecoilState(atom.navSizeState);

    return (
        <Box
            width="80%" 
            alignContent="center"
        >        
        <Tabs isFitted >
            <TabList>
                <Tab>Upscaler</Tab>
                {/* <Tab>Remove Background</Tab> */}
                {/* <Tab>Image to Prompt</Tab> CLIP requires RUST, probably not worth the install unless on cloud */} 
            </TabList>

            <TabPanels>
                <TabPanel>
                    <Upscale></Upscale>
                </TabPanel>
                {/* <TabPanel>
                <p>Placeholder</p>
                </TabPanel> */}
                {/* <TabPanel>
                <p>three!</p>
                </TabPanel> */}
            </TabPanels>
            </Tabs>
        </Box>
    );
}
