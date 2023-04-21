import React from 'react';
import {
    Box,
    Tab,
    TabList,
    TabPanel,
    TabPanels,
    Tabs
} from '@chakra-ui/react';
import Upscale from './Upscale';

export default function ImageEditor () {
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
                    <Upscale />
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
