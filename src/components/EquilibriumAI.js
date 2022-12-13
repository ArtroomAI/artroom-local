import React from "react";
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import EquilibriumLogo from '../images/equilibriumai.png'
import {
    MenuButton,
    HStack,
    Image,
    Text,
    Flex,
    Menu,
    Link,
    Tooltip
} from '@chakra-ui/react'
const EquilibriumAI = () => {
  const [navSize, changeNavSize] = useRecoilState(atom.navSizeState);
  return (
    <>
        <Flex
            mt={15}
            flexDir="column"
            w="100%"
            alignItems={navSize ==="small" ? "center" : "flex-start"}
            onClick={window['openEquilibrium']}
        >
            <Menu placement="right">
                <Link
                    p={2.5}
                    borderRadius={8}
                    _hover={{ textDecor: 'none', backgroundColor: "#AEC8CA" }}
                    >
                    <Tooltip shouldWrapChildren  placement='bottom' label={navSize === "small" ? "Learn More" : ''} fontSize='md'>
                    <MenuButton className="equilibrium-link" bg="transparent" width="100%" >
                        <HStack>
                            <Image width="25px" justify="center" src={EquilibriumLogo} fontSize="xl" color="#82AAAD" />
                            <Text align="center" pl={5} pr={10} fontSize="m" display={navSize === "small" ? "none" : "flex"}>Learn More</Text>
                        </HStack>
                        </MenuButton>
                    </Tooltip>
                </Link>
            </Menu>
        </Flex>
    </>
  );
};

export default EquilibriumAI;