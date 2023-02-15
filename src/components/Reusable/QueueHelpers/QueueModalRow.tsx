import { Flex, Td, Text, Tr } from '@chakra-ui/react';
import React from 'react';

export const QueueModalRow: React.FC<{
	name: string;
	value: string;
}> = ({ name, value }) => (
	<Tr>
		<Td border="none" borderBottomColor="gray.600">
			<Flex align="center" flexWrap="nowrap" minWidth="100%" py=".8rem">
				<Text fontSize="sm" fontWeight="normal" minWidth="100%">
					{name}
				</Text>
			</Flex>
		</Td>

		<Td
			border="none"
			borderBottomColor="gray.600"
			minWidth={{ sm: '250px' }}
			ps="0px">
			<Flex align="center" flexWrap="nowrap" minWidth="100%" py=".8rem">
				<Text fontSize="sm" fontWeight="normal" minWidth="100%">
					{value}
				</Text>
			</Flex>
		</Td>
	</Tr>
);
