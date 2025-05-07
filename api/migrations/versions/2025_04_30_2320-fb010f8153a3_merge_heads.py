"""merge heads

Revision ID: fb010f8153a3
Revises: 5977a0565e82, 6a9f914f656c
Create Date: 2025-04-30 23:20:23.207934

"""
from alembic import op
import models as models
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'fb010f8153a3'
down_revision = ('5977a0565e82', '6a9f914f656c')
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
