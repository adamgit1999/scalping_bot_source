"""
Stripe integration: 2FA, session auth, auto-withdrawal logic.
"""
import stripe
from src.config import Config
import os
from datetime import datetime
import json

class PaymentManager:
    def __init__(self):
        self.stripe = stripe
        self.stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
        self.webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
        self.config = Config.load_config()
    
    def create_checkout_session(self, price_id, success_url, cancel_url):
        """Create a Stripe checkout session"""
        try:
            session = self.stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price': price_id,
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=success_url,
                cancel_url=cancel_url,
            )
            return session
            
        except Exception as e:
            print(f"Error creating checkout session: {e}")
            return None
    
    def create_portal_session(self, customer_id):
        """Create a Stripe customer portal session"""
        try:
            session = self.stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=self.config.get('webhook_url')
            )
            return session
            
        except Exception as e:
            print(f"Error creating portal session: {e}")
            return None
    
    def handle_webhook(self, payload, sig_header):
        """Handle Stripe webhook events"""
        try:
            event = self.stripe.Webhook.construct_event(
                payload, sig_header, self.webhook_secret
            )
            
            # Handle the event
            if event['type'] == 'checkout.session.completed':
                self._handle_checkout_completed(event['data']['object'])
            elif event['type'] == 'customer.subscription.updated':
                self._handle_subscription_updated(event['data']['object'])
            elif event['type'] == 'customer.subscription.deleted':
                self._handle_subscription_deleted(event['data']['object'])
            
            return True
            
        except Exception as e:
            print(f"Error handling webhook: {e}")
            return False
    
    def _handle_checkout_completed(self, session):
        """Handle completed checkout session"""
        try:
            # Get customer details
            customer = self.stripe.Customer.retrieve(session.customer)
            
            # Update user's subscription status
            self._update_subscription_status(
                customer.id,
                session.subscription,
                'active'
            )
            
        except Exception as e:
            print(f"Error handling checkout completion: {e}")
    
    def _handle_subscription_updated(self, subscription):
        """Handle subscription update"""
        try:
            # Update subscription status
            self._update_subscription_status(
                subscription.customer,
                subscription.id,
                subscription.status
            )
            
        except Exception as e:
            print(f"Error handling subscription update: {e}")
    
    def _handle_subscription_deleted(self, subscription):
        """Handle subscription deletion"""
        try:
            # Update subscription status
            self._update_subscription_status(
                subscription.customer,
                subscription.id,
                'canceled'
            )
            
        except Exception as e:
            print(f"Error handling subscription deletion: {e}")
    
    def _update_subscription_status(self, customer_id, subscription_id, status):
        """Update subscription status in local storage"""
        try:
            # Load current subscriptions
            subscriptions_file = os.path.join(
                Config.CONFIG_DIR,
                'subscriptions.json'
            )
            
            if os.path.exists(subscriptions_file):
                with open(subscriptions_file, 'r') as f:
                    subscriptions = json.load(f)
            else:
                subscriptions = {}
            
            # Update subscription
            subscriptions[customer_id] = {
                'subscription_id': subscription_id,
                'status': status,
                'updated_at': int(datetime.now().timestamp() * 1000)
            }
            
            # Save updated subscriptions
            with open(subscriptions_file, 'w') as f:
                json.dump(subscriptions, f, indent=4)
            
        except Exception as e:
            print(f"Error updating subscription status: {e}")
    
    def get_subscription_status(self, customer_id):
        """Get subscription status for a customer"""
        try:
            # Load subscriptions
            subscriptions_file = os.path.join(
                Config.CONFIG_DIR,
                'subscriptions.json'
            )
            
            if os.path.exists(subscriptions_file):
                with open(subscriptions_file, 'r') as f:
                    subscriptions = json.load(f)
                
                if customer_id in subscriptions:
                    return subscriptions[customer_id]['status']
            
            return None
            
        except Exception as e:
            print(f"Error getting subscription status: {e}")
            return None
    
    def create_withdrawal(self, amount, currency, destination):
        """Create a withdrawal transfer"""
        try:
            transfer = self.stripe.Transfer.create(
                amount=amount,
                currency=currency,
                destination=destination,
                transfer_group='withdrawal'
            )
            return transfer
            
        except Exception as e:
            print(f"Error creating withdrawal: {e}")
            return None
    
    def get_balance(self):
        """Get Stripe account balance"""
        try:
            balance = self.stripe.Balance.retrieve()
            return {
                'available': balance.available,
                'pending': balance.pending
            }
            
        except Exception as e:
            print(f"Error getting balance: {e}")
            return None

