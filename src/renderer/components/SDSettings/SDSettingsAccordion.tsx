import React from 'react';
import { Accordion, AccordionItem, AccordionButton, AccordionIcon, AccordionPanel, Box } from "@chakra-ui/react";

export const SDSettingsAccordion = ({ header, children } : { header: string; children: React.ReactNode }) => {
    return (
        <Accordion allowToggle={true} border="none" bg="transparent" width="100%">
           <AccordionItem border="none">
                {({ isExpanded }) => (
                    <>
                        <AccordionButton p={0} bg="transparent" _hover={{ bg: 'transparent' }}>
                            <Box width="100%" flex="1" textAlign="start">
                                <h1><b>{ header }</b></h1>
                            </Box>
                            <AccordionIcon />
                        </AccordionButton>
                        <AccordionPanel p={0} mt={4} mb={2} width="100%" bg="transparent">
                            { isExpanded && children }
                        </AccordionPanel>
                    </>
                )}
            </AccordionItem>
        </Accordion>
    );
}