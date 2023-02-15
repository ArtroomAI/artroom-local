import React from 'react';
import {
	Center,
	Text,
	useColorModeValue,
	Link,
	HStack,
	chakra,
} from '@chakra-ui/react';
import { Link as RouterLink } from 'react-router-dom';

export const Footer: React.FC = () => (
	<Center
		py={{ base: 2 }}
		minH={'60px'}
		borderTop={1}
		borderStyle="solid"
		borderColor={useColorModeValue('gray.200', 'gray.900')}>
		<HStack>
			<Text>© 2022 Artroom AI</Text>
			<chakra.span>&#183;</chakra.span>
			<Link as={RouterLink} to="/privacy-policy">
				Privacy Policy
			</Link>
		</HStack>
	</Center>
);
