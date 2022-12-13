import React from "react";
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import {FaDiscord} from 'react-icons/fa'
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
const Discord = () => {
  const [navSize, changeNavSize] = useRecoilState(atom.navSizeState);
  return (
    <>
        <Flex
            mt={15}
            flexDir="column"
            w="100%"
            alignItems={navSize ==="small" ? "center" : "flex-start"}
            onClick={window['openDiscord']}
        >
            <Menu placement="right">
                <Link
                    p={2.5}
                    borderRadius={8}
                    _hover={{ textDecor: 'none', backgroundColor: "#AEC8CA" }}
                    >
                    <Tooltip shouldWrapChildren  placement='bottom' label={navSize === "small" ? "Join Discord" : ''} fontSize='md'>
                    <MenuButton className="discord-link" bg="transparent" width="100%" >
                        <HStack>
                            <Icon justify="center" as={FaDiscord} fontSize="xl" color="#82AAAD" />
                            <Text align="center" pl={5} pr={10} fontSize="m" display={navSize === "small" ? "none" : "flex"}>Join Discord</Text>
                        </HStack>
                        </MenuButton>
                    </Tooltip>
                </Link>
            </Menu>
        </Flex>
    </>
  );
};

export default Discord;