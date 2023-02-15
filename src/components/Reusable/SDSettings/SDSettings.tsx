import React from 'react';
import { Tabs, TabList, Tab, TabPanels, TabPanel, Box } from '@chakra-ui/react';
import { Layers, Parameters } from './tabs';

interface ISDSettingsProps {
	showLayersTab: boolean;
}

export const SDSettings: React.FC<ISDSettingsProps> = ({ showLayersTab }) => {
	if (!showLayersTab) {
		return (
			<Box w="400px" p={4}>
				<Parameters />
			</Box>
		);
	}

	return (
		<Box w="400px">
			<Tabs p={4} h="100%" isLazy>
				<TabList>
					<Tab>Parameters</Tab>
					<Tab>Layers</Tab>
				</TabList>

				<TabPanels h="100%">
					<TabPanel p="0" pt={4}>
						<Parameters />
					</TabPanel>
					<TabPanel p="0" h="100%" pt={4}>
						<Layers />
					</TabPanel>
				</TabPanels>
			</Tabs>
		</Box>
	);
};
