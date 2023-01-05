// ********************* Bare interrupt handlers *********************
// Interrupts for STM32H7x5

void WWDG_IRQHandler(void) {handle_interrupt(WWDG_IRQn);}
void PVD_AVD_IRQHandler(void) {handle_interrupt(PVD_AVD_IRQn);}
void TAMP_STAMP_IRQHandler(void) {handle_interrupt(TAMP_STAMP_IRQn);}
void RTC_WKUP_IRQHandler(void) {handle_interrupt(RTC_WKUP_IRQn);}
void FLASH_IRQHandler(void) {handle_interrupt(FLASH_IRQn);}
void RCC_IRQHandler(void) {handle_interrupt(RCC_IRQn);}
void EXTI0_IRQHandler(void) {handle_interrupt(EXTI0_IRQn);}
void EXTI1_IRQHandler(void) {handle_interrupt(EXTI1_IRQn);}
void EXTI2_IRQHandler(void) {handle_interrupt(EXTI2_IRQn);}
void EXTI3_IRQHandler(void) {handle_interrupt(EXTI3_IRQn);}
void EXTI4_IRQHandler(void) {handle_interrupt(EXTI4_IRQn);}
void DMA1_Stream0_IRQHandler(void) {handle_interrupt(DMA1_Stream0_IRQn);}
void DMA1_Stream1_IRQHandler(void) {handle_interrupt(DMA1_Stream1_IRQn);}
void DMA1_Stream2_IRQHandler(void) {handle_interrupt(DMA1_Stream2_IRQn);}
void DMA1_Stream3_IRQHandler(void) {handle_interrupt(DMA1_Stream3_IRQn);}
void DMA1_Stream4_IRQHandler(void) {handle_interrupt(DMA1_Stream4_IRQn);}
void DMA1_Stream5_IRQHandler(void) {handle_interrupt(DMA1_Stream5_IRQn);}
void DMA1_Stream6_IRQHandler(void) {handle_interrupt(DMA1_Stream6_IRQn);}
void ADC_IRQHandler(void) {handle_interrupt(ADC_IRQn);}
void EXTI9_5_IRQHandler(void) {handle_interrupt(EXTI9_5_IRQn);}
void TIM1_BRK_IRQHandler(void) {handle_interrupt(TIM1_BRK_IRQn);}
void TIM1_UP_TIM10_IRQHandler(void) {handle_interrupt(TIM1_UP_TIM10_IRQn);}
void TIM1_TRG_COM_IRQHandler(void) {handle_interrupt(TIM1_TRG_COM_IRQn);}
void TIM1_CC_IRQHandler(void) {handle_interrupt(TIM1_CC_IRQn);}
void TIM2_IRQHandler(void) {handle_interrupt(TIM2_IRQn);}
void TIM3_IRQHandler(void) {handle_interrupt(TIM3_IRQn);}
void TIM4_IRQHandler(void) {handle_interrupt(TIM4_IRQn);}
void I2C1_EV_IRQHandler(void) {handle_interrupt(I2C1_EV_IRQn);}
void I2C1_ER_IRQHandler(void) {handle_interrupt(I2C1_ER_IRQn);}
void I2C2_EV_IRQHandler(void) {handle_interrupt(I2C2_EV_IRQn);}
void I2C2_ER_IRQHandler(void) {handle_interrupt(I2C2_ER_IRQn);}
void SPI1_IRQHandler(void) {handle_interrupt(SPI1_IRQn);}
void SPI2_IRQHandler(void) {handle_interrupt(SPI2_IRQn);}
void USART1_IRQHandler(void) {handle_interrupt(USART1_IRQn);}
void USART2_IRQHandler(void) {handle_interrupt(USART2_IRQn);}
void USART3_IRQHandler(void) {handle_interrupt(USART3_IRQn);}
void EXTI15_10_IRQHandler(void) {handle_interrupt(EXTI15_10_IRQn);}
void RTC_Alarm_IRQHandler(void) {handle_interrupt(RTC_Alarm_IRQn);}
void TIM8_BRK_TIM12_IRQHandler(void) {handle_interrupt(TIM8_BRK_TIM12_IRQn);}
void TIM8_UP_TIM13_IRQHandler(void) {handle_interrupt(TIM8_UP_TIM13_IRQn);}
void TIM8_TRG_COM_TIM14_IRQHandler(void) {handle_interrupt(TIM8_TRG_COM_TIM14_IRQn);}
void TIM8_CC_IRQHandler(void) {handle_interrupt(TIM8_CC_IRQn);}
void DMA1_Stream7_IRQHandler(void) {handle_interrupt(DMA1_Stream7_IRQn);}
void TIM5_IRQHandler(void) {handle_interrupt(TIM5_IRQn);}
void SPI3_IRQHandler(void) {handle_interrupt(SPI3_IRQn);}
void UART4_IRQHandler(void) {handle_interrupt(UART4_IRQn);}
void UART5_IRQHandler(void) {handle_interrupt(UART5_IRQn);}
void TIM6_DAC_IRQHandler(void) {handle_interrupt(TIM6_DAC_IRQn);}
void TIM7_IRQHandler(void) {handle_interrupt(TIM7_IRQn);}
void DMA2_Stream0_IRQHandler(void) {handle_interrupt(DMA2_Stream0_IRQn);}
void DMA2_Stream1_IRQHandler(void) {handle_interrupt(DMA2_Stream1_IRQn);}
void DMA2_Stream2_IRQHandler(void) {handle_interrupt(DMA2_Stream2_IRQn);}
void DMA2_Stream3_IRQHandler(void) {handle_interrupt(DMA2_Stream3_IRQn);}
void DMA2_Stream4_IRQHandler(void) {handle_interrupt(DMA2_Stream4_IRQn);}
void DMA2_Stream5_IRQHandler(void) {handle_interrupt(DMA2_Stream5_IRQn);}
void DMA2_Stream6_IRQHandler(void) {handle_interrupt(DMA2_Stream6_IRQn);}
void DMA2_Stream7_IRQHandler(void) {handle_interrupt(DMA2_Stream7_IRQn);}
void USART6_IRQHandler(void) {handle_interrupt(USART6_IRQn);}
void I2C3_EV_IRQHandler(void) {handle_interrupt(I2C3_EV_IRQn);}
void I2C3_ER_IRQHandler(void) {handle_interrupt(I2C3_ER_IRQn);}
void FDCAN1_IT0_IRQHandler(void) {handle_interrupt(FDCAN1_IT0_IRQn);}
void FDCAN1_IT1_IRQHandler(void) {handle_interrupt(FDCAN1_IT1_IRQn);}
void FDCAN2_IT0_IRQHandler(void) {handle_interrupt(FDCAN2_IT0_IRQn);}
void FDCAN2_IT1_IRQHandler(void) {handle_interrupt(FDCAN2_IT1_IRQn);}
void FDCAN3_IT0_IRQHandler(void) {handle_interrupt(FDCAN3_IT0_IRQn);}
void FDCAN3_IT1_IRQHandler(void) {handle_interrupt(FDCAN3_IT1_IRQn);}
void FDCAN_CAL_IRQHandler(void) {handle_interrupt(FDCAN_CAL_IRQn);}
void OTG_HS_EP1_OUT_IRQHandler(void) {handle_interrupt(OTG_HS_EP1_OUT_IRQn);}
void OTG_HS_EP1_IN_IRQHandler(void) {handle_interrupt(OTG_HS_EP1_IN_IRQn);}
void OTG_HS_WKUP_IRQHandler(void) {handle_interrupt(OTG_HS_WKUP_IRQn);}
void OTG_HS_IRQHandler(void) {handle_interrupt(OTG_HS_IRQn);}
void UART7_IRQHandler(void) {handle_interrupt(UART7_IRQn);}
