import React from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import {FaFileImage} from 'react-icons/fa'
import {
    MenuButton,
    HStack,
    Icon,
    Text,
    Flex,
    Menu,
    Link,
    Tooltip
} from '@chakra-ui/react'
// Tour component
const Viewer = () => {
  // Tour state is the state which control the JoyRide component
  const [navSize, changeNavSize] = useRecoilState(atom.navSizeState);
  return (
    <>
        <Flex
            mt={25}
            flexDir='column'
            w='100%'
            alignItems={navSize ==='small' ? 'center' : 'flex-start'}
        >
            <Menu placement='right'>
                <Link
                    p={2.5}
                    borderRadius={8}
                    _hover={{ textDecor: 'none', backgroundColor: '#AEC8CA' }}
                    >
                    <Tooltip shouldWrapChildren  placement='bottom' label={navSize === 'small' ? 'View Images' : ''} fontSize='md'>
                    <MenuButton className='viewer-link' bg='transparent' width='100%' onClick={window['getImageDir']}>
                        <HStack>
                            <Icon justify='center' as={FaFileImage} fontSize='xl' color='#82AAAD' />
                            <Text align='center' pl={5} pr={10} fontSize='m' display={navSize === 'small' ? 'none' : 'flex'}>View Images</Text>
                        </HStack>
                        </MenuButton>
                    </Tooltip>
                </Link>
            </Menu>
        </Flex>
    </>
  );
};

export default Viewer;